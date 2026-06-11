"""Metadata constant injected into the Jupyter kernel."""

from typing import Any, Dict, List, Optional


class Metadata:
    """Input metadata: video FPS, frame count, duration, etc.

    Single-video / multi-image samples populate ``fps``, ``total_frames``,
    ``duration_sec``, ``num_images`` (and leave ``videos=None``,
    ``num_videos=1``).

    Multi-video samples populate ``videos`` — a list of dicts, one per
    backing video, with the keys ``name``, ``fps``, ``num_frames``,
    ``duration_sec`` — and ``num_videos > 1``. In multi-video mode the
    kernel exposes ``InputImages_1, InputImages_2, ...`` (one per video,
    1-indexed) instead of a concatenated ``InputImages``; ``Metadata.videos[i-1]``
    describes ``InputImages_<i>``.
    """

    def __init__(
        self,
        is_video: bool,
        fps: Optional[float] = None,
        total_frames: Optional[int] = None,
        duration_sec: Optional[float] = None,
        num_images: Optional[int] = None,
        video_source: Optional[str] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
        num_videos: int = 1,
        **extra: Any,
    ):
        self.is_video = is_video
        self.fps = fps
        self.total_frames = total_frames
        self.duration_sec = duration_sec
        self.num_images = num_images
        self.video_source = video_source
        self.videos = videos
        self.num_videos = num_videos
        # Store any extra metadata fields
        for k, v in extra.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self) -> str:
        parts = [f"is_video={self.is_video}"]
        if self.num_videos > 1:
            parts.append(f"num_videos={self.num_videos}")
            if self.videos:
                names = [v.get("name", f"video_{i+1}") for i, v in enumerate(self.videos)]
                parts.append(f"videos={names}")
            return f"Metadata({', '.join(parts)})"
        if self.fps is not None:
            parts.append(f"fps={self.fps}")
        if self.total_frames is not None:
            parts.append(f"total_frames={self.total_frames}")
        if self.duration_sec is not None:
            parts.append(f"duration_sec={self.duration_sec:.1f}")
        if self.num_images is not None:
            parts.append(f"num_images={self.num_images}")
        return f"Metadata({', '.join(parts)})"
