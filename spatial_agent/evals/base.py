"""Base benchmark and sample classes."""

import errno
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaseBenchmarkSample:
    """A single benchmark sample.

    For multi-video samples, populate ``image_groups`` (list of per-video
    frame paths) and ``frame_indices_groups``; the workflow then injects
    ``InputImages_1, InputImages_2, ...`` into the kernel instead of a
    single concatenated ``InputImages``. Single-video / multi-image
    samples leave these as ``None`` and the workflow uses ``images``.
    """

    sample_id: Any
    question: str
    question_type: str
    images: List[str]  # file paths or URLs (concat across videos for multi-video)
    answer: str
    # Multi-video fields (None for single-video / multi-image samples)
    image_groups: Optional[List[List[str]]] = field(default=None)
    frame_indices_groups: Optional[List[List[int]]] = field(default=None)
    fps_per_video: Optional[List[float]] = field(default=None)
    total_frames_per_video: Optional[List[int]] = field(default=None)
    duration_per_video: Optional[List[float]] = field(default=None)
    video_names: Optional[List[str]] = field(default=None)
    # Per-video source paths (None for samples with frames-only multi-video,
    # e.g. MMSI-Video where the original videos aren't shipped). Populated by
    # CVBench so SAM3.segment_video_*(..., video_index=N) can decode video N.
    video_sources_per_video: Optional[List[str]] = field(default=None)


@dataclass
class LazyVideoSample(BaseBenchmarkSample):
    """Benchmark sample whose frames are lazily extracted from a backing video."""

    video: str = ""
    frame_indices: List[int] = field(default_factory=list)
    fps: float = 0.0
    total_video_frames: int = 0
    duration_sec: float = 0.0
    _bench_ref: Any = field(default=None, repr=False)
    _frames_loaded: bool = field(default=False, repr=False)

    def ensure_frames_loaded(self) -> None:
        """Extract frames on first access using the owning benchmark."""
        if self._frames_loaded:
            return
        if self._bench_ref is None or not self.video:
            self._frames_loaded = True
            return
        frames, frame_indices, fps, total = self._bench_ref._extract_frames(self.video)
        self.images = frames
        self.frame_indices = frame_indices
        self.fps = fps
        self.total_video_frames = total
        self.duration_sec = total / fps if fps else 0.0
        self._frames_loaded = True


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    data_specific_prompt: str = ""

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self.data_path = data_path
        self.question_type_filter = question_type
        self.data: List[BaseBenchmarkSample] = []
        self.read_data()

    @abstractmethod
    def read_data(self) -> None:
        """Populate self.data with samples."""
        ...

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> BaseBenchmarkSample:
        return self.data[index]

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> BaseBenchmarkSample:
        if self._iter_idx >= len(self.data):
            raise StopIteration
        sample = self.data[self._iter_idx]
        self._iter_idx += 1
        return sample

    def extract_answer(self, prediction: str) -> str:
        """Extract a structured answer from raw prediction text."""
        if not prediction:
            return ""

        # Strategy 1: \boxed{X}
        m = re.search(r"\\boxed\{([A-Za-z])\}", prediction)
        if m:
            return m.group(1).upper()

        # Strategy 2: (A), A., A:
        m = re.search(r"\(?([A-Za-z])\)?[\.\:\)]", prediction)
        if m:
            return m.group(1).upper()

        # Strategy 3: single letter
        clean = re.sub(r"[^A-Za-z]", "", prediction)
        if len(clean) == 1:
            return clean.upper()

        # Strategy 4: return as-is
        return prediction.strip()

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        """Evaluate a single prediction against its ground truth.

        Returns a score in [0.0, 1.0] where 1.0 = correct, 0.0 = incorrect,
        and intermediate values represent partial credit (e.g. MRA).
        Returns None if the benchmark cannot score this sample (e.g. needs
        external tools like LLM judge).

        Default: letter-based MC extraction and exact match.
        Subclasses should override for dataset-specific evaluation.
        """
        pred = self.extract_answer(prediction)
        gt = self.extract_answer(sample.answer)
        return 1.0 if (pred and pred == gt) else 0.0

    @abstractmethod
    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate predictions against ground truth."""
        ...

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        """Print human-readable evaluation summary."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.__class__.__name__}")
        print(f"Total: {results.get('total_samples', '?')}")
        print(f"Correct: {results.get('correct_samples', '?')}")
        print(f"Accuracy: {results.get('overall_accuracy', 0):.4f}")
        print(f"{'='*60}\n")


def default_video_frame_cache_dir(video_path: str) -> str:
    """Return the standard on-disk cache location for extracted frames."""
    return os.path.join(
        os.path.dirname(video_path),
        ".frame_cache",
        os.path.basename(video_path),
    )


def fallback_video_frame_cache_dir(video_path: str) -> str:
    """Return a writable fallback cache directory for read-only datasets."""
    digest = hashlib.sha1(os.path.abspath(video_path).encode("utf-8")).hexdigest()
    return os.path.join(
        tempfile.gettempdir(),
        "spatial_agent_frame_cache",
        digest,
    )


class VideoFrameBenchmarkMixin:
    """Shared ffmpeg-backed frame extraction for video benchmarks."""

    _config: Any

    def _extract_frames(self, video_path: str):
        return extract_video_frames(
            video_path,
            default_video_frame_cache_dir(video_path),
            video_max_fps=getattr(self._config, "video_max_fps", None),
            video_frame_resize_short_edge=getattr(
                self._config, "video_frame_resize_short_edge", None
            ),
        )


def write_bytes_if_missing(path: str, payload: bytes) -> str:
    """Persist binary payload to disk if it is not already cached."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
    return path


def save_embedded_image(
    image_path: str,
    image_data: Any,
    *,
    convert_rgb: bool = False,
) -> str:
    """Persist an embedded image payload or PIL image to disk."""
    if os.path.exists(image_path):
        return image_path

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    if isinstance(image_data, dict) and "bytes" in image_data:
        image_data = image_data["bytes"]

    if isinstance(image_data, (bytes, bytearray)):
        if convert_rgb:
            from PIL import Image

            with Image.open(io.BytesIO(image_data)) as img:
                img.convert("RGB").save(image_path)
        else:
            write_bytes_if_missing(image_path, bytes(image_data))
        return image_path

    if hasattr(image_data, "save"):
        image = image_data.convert("RGB") if convert_rgb and hasattr(image_data, "convert") else image_data
        image.save(image_path)
        return image_path

    raise TypeError(f"Unsupported image payload type: {type(image_data)!r}")


def _find_ffmpeg() -> str:
    """Find ffmpeg binary: system PATH first, then imageio_ffmpeg fallback."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # last resort, will fail with clear error


def _probe_video_meta(ffmpeg: str, video_path: str):
    """Get fps, total_frames, duration using ffmpeg -i (no ffprobe needed)."""
    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True, timeout=30,
    )
    # ffmpeg -i writes info to stderr
    info = result.stderr

    # Parse fps: "30 fps" or "29.97 fps"
    import re as _re
    fps = 30.0
    m = _re.search(r"(\d+(?:\.\d+)?)\s+fps", info)
    if m:
        fps = float(m.group(1))

    # Parse duration: "Duration: 00:04:00.00"
    duration = 0.0
    m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    total = int(duration * fps) if duration else 0
    return fps, total, duration


_FRAME_CACHE_LOCK_TIMEOUT = 600


@contextmanager
def _locked_frame_cache(cache_dir: str, video_path: str):
    """Lock a writable cache and own one private extraction directory."""
    fallback = fallback_video_frame_cache_dir(video_path)
    for candidate in dict.fromkeys((cache_dir, fallback)):
        lock_file = None
        staging_dir = None
        try:
            try:
                os.makedirs(candidate, exist_ok=True)
                # Keep this file permanently: replacing it would split the lock.
                lock_file = open(os.path.join(candidate, ".lock"), "a")
                deadline = time.monotonic() + _FRAME_CACHE_LOCK_TIMEOUT
                while True:
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EAGAIN, errno.EACCES):
                            raise
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"Timed out waiting for frame cache: {candidate}")
                        time.sleep(min(0.05, remaining))
                # Also detects an existing cache with an unwritable directory.
                staging_dir = tempfile.mkdtemp(prefix=".staging-", dir=candidate)
            except OSError as exc:
                if candidate != fallback and exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
                    continue
                raise
            yield candidate, staging_dir
            return
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            if lock_file is not None:
                lock_file.close()


def _frame_cache_identity(video_path, video_max_fps, video_frame_resize_short_edge):
    source = os.stat(video_path)
    return {
        "version": 1,
        "source": os.path.realpath(video_path),
        "size": source.st_size,
        "mtime_ns": source.st_mtime_ns,
        "ctime_ns": source.st_ctime_ns,
        "video_max_fps": video_max_fps,
        "video_frame_resize_short_edge": video_frame_resize_short_edge,
    }


def _frame_file_stat(path):
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _validate_frame_images(frame_paths):
    from PIL import Image

    for path in frame_paths:
        with Image.open(path) as image:
            image.load()


def _read_frame_cache(cache_dir, identity):
    """Read complete metadata; malformed, stale, or damaged caches are misses."""
    try:
        with open(os.path.join(cache_dir, "meta.json")) as file:
            meta = json.load(file)
        if not isinstance(meta, dict):
            return None
        if "cache_identity" in meta and meta["cache_identity"] != identity:
            return None
        files = meta["frame_files"]
        indices = meta["frame_indices"]
        fps = meta["fps"]
        if not isinstance(files, list) or not files or not all(isinstance(name, str) for name in files):
            return None
        if not isinstance(indices, list) or len(indices) != len(files):
            return None
        if any(type(index) is not int or index < 0 for index in indices):
            return None
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            return None
        total = meta.get("total_video_frames", len(files))
        if type(total) is not int or total < 0:
            return None
        frame_paths = [os.path.join(cache_dir, name) for name in files]
        cache_root = os.path.realpath(cache_dir)
        if any(os.path.isabs(name) for name in files) or any(
            os.path.commonpath((cache_root, os.path.realpath(path))) != cache_root
            for path in frame_paths
        ):
            return None
        if "cache_identity" in meta:
            # These immutable files were decoded before publication. Stat checks
            # detect later damage without decoding a whole video on every hit.
            if meta.get("frame_stats") != [_frame_file_stat(path) for path in frame_paths]:
                return None
        else:
            # Legacy metadata has neither settings identity nor a verified manifest.
            _validate_frame_images(frame_paths)
        return frame_paths, indices, fps, total
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def extract_video_frames(
    video_path: str,
    cache_dir: str,
    video_max_fps: Optional[float] = None,
    video_frame_resize_short_edge: Optional[int] = None,
) -> Tuple[List[str], List[int], float, int]:
    """Extract frames from a video using ffmpeg (codec-agnostic).

    Handles H.264, HEVC, AV1, VP9, and all other codecs supported by ffmpeg.
    Uses imageio_ffmpeg as fallback when system ffmpeg is not on PATH.
    Results are cached in immutable generations, published atomically under a
    per-cache lock. Empty or failed extractions are never published.

    Args:
        video_path: Path to the video file.
        cache_dir: Directory for frame cache.
        video_max_fps: Max output fps (None = native fps).
        video_frame_resize_short_edge: Resize short edge to this value (None = no resize).

    Returns:
        (frame_paths, frame_indices, fps, total_video_frames)
    """
    if not os.path.exists(video_path):
        print(f"[Warning] Video not found: {video_path}")
        return [], [], 0.0, 0

    # Published files are immutable, so hits need no writable lock or directory.
    # Recheck under the lock after a miss, since another producer may finish first.
    identity = _frame_cache_identity(video_path, video_max_fps, video_frame_resize_short_edge)
    cached = _read_frame_cache(cache_dir, identity)
    if cached is not None:
        return cached

    with _locked_frame_cache(cache_dir, video_path) as (cache_dir, staging_dir):
        identity = _frame_cache_identity(video_path, video_max_fps, video_frame_resize_short_edge)
        cached = _read_frame_cache(cache_dir, identity)
        if cached is not None:
            return cached

        ffmpeg = _find_ffmpeg()
        fps, total, duration = _probe_video_meta(ffmpeg, video_path)

        # Preserve the extraction filters, JPEG quality, and frame-index mapping.
        if video_max_fps and fps > video_max_fps:
            out_fps = video_max_fps
        else:
            out_fps = fps
        vf_filters = [f"fps={out_fps}"]
        if video_frame_resize_short_edge:
            se = video_frame_resize_short_edge
            vf_filters.append(
                f"scale='if(lte(iw,ih),{se},-2)':'if(lte(iw,ih),-2,{se})'"
            )
        cmd = [
            ffmpeg, "-v", "error",
            "-i", video_path,
            "-vf", ",".join(vf_filters),
            "-q:v", "2",
            os.path.join(staging_dir, "frame_%06d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result.check_returncode()

        frame_files = sorted(
            name for name in os.listdir(staging_dir)
            if name.startswith("frame_") and name.endswith(".jpg")
        )
        frame_paths = [os.path.join(staging_dir, name) for name in frame_files]
        step = fps / out_fps
        frame_indices = [int(round(i * step)) for i in range(len(frame_files))]
        if not frame_paths:
            print(f"[Warning] No frames extracted from {video_path}")
            return [], [], fps, total

        _validate_frame_images(frame_paths)
        if identity != _frame_cache_identity(video_path, video_max_fps, video_frame_resize_short_edge):
            raise RuntimeError(f"Video changed during frame extraction: {video_path}")
        generation_name = os.path.basename(staging_dir).replace(".staging-", ".generation-", 1)
        generation_dir = os.path.join(cache_dir, generation_name)
        meta = {
            "frame_indices": frame_indices,
            "frame_files": [os.path.join(generation_name, name) for name in frame_files],
            "fps": fps,
            "total_video_frames": total,
            "cache_identity": identity,
            "frame_stats": [_frame_file_stat(path) for path in frame_paths],
        }
        with open(os.path.join(staging_dir, "meta.json"), "w") as file:
            json.dump(meta, file)
            file.flush()
            os.fsync(file.fileno())

        # Never rewrite/remove an older generation: callers may still be using it.
        os.rename(staging_dir, generation_dir)
        # Keep this generation even if publication raises: the rename may have
        # committed before an interrupt or filesystem error reaches this process.
        os.replace(os.path.join(generation_dir, "meta.json"), os.path.join(cache_dir, "meta.json"))
        return [os.path.join(cache_dir, name) for name in meta["frame_files"]], frame_indices, fps, total
