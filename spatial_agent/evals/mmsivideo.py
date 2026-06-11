"""MMSI-Video-Bench data loader.

Data structure:
    data/MMSI-Video-Bench/mmsivideo.json
    data/MMSI-Video-Bench/frames/
    data/MMSI-Video-Bench/videos/
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_results_summary


CATEGORY_MAP = {
    # Short names (legacy)
    "Planning": "Planning",
    "Prediction": "Planning",
    "Memory Update": "Cross-Video",
    "Multi-View Integration": "Cross-Video",
    "Camera Motion": "Motion Understanding",
    "Instance Motion": "Motion Understanding",
    "Interactive Motion": "Motion Understanding",
    "Camera-Instance": "Spatial Construction",
    "Camera-Scene": "Spatial Construction",
    "Instance-Instance": "Spatial Construction",
    "Instance-Scene": "Spatial Construction",
    "Attribute": "Spatial Construction",
    "Scene-Scene": "Spatial Construction",
    # Full names as they appear in mmsivideo.json
    "(Cross-Video) Memoery Update": "Cross-Video",
    "(Cross-Video) Multi-View Integration": "Cross-Video",
    "(Motion Understanding) Camera Motion": "Motion Understanding",
    "(Motion Understanding) Instance Motion": "Motion Understanding",
    "(Motion Understanding) Interactive Motion": "Motion Understanding",
    "(Spatial Construction) Camera-Instance Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Camera-Scene Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance-Instance Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance-Scene Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance/Scene Attribute": "Spatial Construction",
    "(Spatial Construction) Scene-Scene Spatial Relationship": "Spatial Construction",
}


@dataclass
class MMSIVideoBenchSample(BaseBenchmarkSample):
    """MMSI-Video-Bench sample.

    Multi-video samples populate ``image_groups``/``frame_indices_groups``
    (inherited from BaseBenchmarkSample) so the workflow injects
    ``InputImages_1, InputImages_2, ...`` instead of a flat ``InputImages``.
    Single-video samples leave those fields ``None`` and behave like any
    other single-video benchmark.
    """

    choices: List[str] = field(default_factory=list)
    hint: str = ""
    video: Optional[str] = None  # video file path for SAM3 segment_video_*
    fps: Optional[float] = None
    duration_sec: Optional[float] = None
    total_video_frames: Optional[int] = None
    num_videos: int = 1
    ref_images: List[str] = field(default_factory=list)


class MMSIVideoBench(BaseBenchmark):
    """MMSI-Video-Bench loader."""

    data_specific_prompt = (
        "Select the best answer from the given options. "
        "Answer with a single letter corresponding to the correct choice."
    )

    def read_data(self) -> None:
        json_path = os.path.join(self.data_path, "mmsivideo.json")
        if not os.path.exists(json_path):
            print(f"[Warning] MMSI-Video-Bench JSON not found at {json_path}")
            return

        with open(json_path, "r") as f:
            items = json.load(f)

        for item in items:
            qtype = item.get("type", "")
            if self.question_type_filter and qtype not in self.question_type_filter:
                continue

            # Per-video frame paths — preserve grouping for multi-video samples
            frames_list = item.get("frames_list", [[]])
            groups: List[List[str]] = [
                [os.path.join(self.data_path, "frames", p) for p in group]
                for group in frames_list
            ]
            full_paths: List[str] = [p for g in groups for p in g]

            # Video metadata (fps, duration) from video_list
            video_list = item.get("video_list", [])
            if video_list:
                # Use first video's fps (all videos in a sample share the same fps)
                fps = video_list[0].get("base_fps")
                # Total duration = sum of all videos' durations
                total_duration = 0.0
                for v in video_list:
                    start = v.get("start", 0.0)
                    end = v.get("end", 0.0)
                    total_duration += end - start
                duration_sec = total_duration if total_duration > 0 else None
            else:
                fps = None
                duration_sec = None

            options = item.get("options", [])
            gt = item.get("ground_truth", "")
            num_videos = len(frames_list)

            # Per-video metadata used for InputImages_N annotation in multi-video mode.
            if num_videos > 1:
                if video_list and len(video_list) == num_videos:
                    fps_per_video = [v.get("base_fps") or fps or 0.0 for v in video_list]
                    duration_per_video = [
                        max(0.0, (v.get("end", 0.0) or 0.0) - (v.get("start", 0.0) or 0.0))
                        for v in video_list
                    ]
                    video_names = [
                        os.path.basename(v.get("path", "") or f"video_{i+1}")
                        for i, v in enumerate(video_list)
                    ]
                else:
                    fps_per_video = [fps or 0.0] * num_videos
                    duration_per_video = [0.0] * num_videos
                    video_names = [f"video_{i+1}" for i in range(num_videos)]
                total_frames_per_video = [len(g) for g in groups]
                # Per-video frame indices: each video's frames are 0-indexed locally
                frame_indices_groups = [list(range(len(g))) for g in groups]
                image_groups = groups
            else:
                fps_per_video = None
                duration_per_video = None
                video_names = None
                total_frames_per_video = None
                frame_indices_groups = None
                image_groups = None

            # Video file path for SAM3 segment_video_* (single-video only;
            # multi-video samples use segment_image_* on pre-extracted frames)
            video_path = None
            if len(video_list) == 1:
                video_path = os.path.join(
                    self.data_path, "videos", video_list[0]["path"]
                )

            # Reference images (inline <image> tags in the question).
            raw_refs = item.get("ref_images") or []
            ref_rel_paths = [p for p in raw_refs if isinstance(p, str)]
            ref_full_paths = [
                os.path.join(self.data_path, "ref_images", p)
                for p in ref_rel_paths
            ]

            question = item.get("ori_question", "")
            if ref_full_paths:
                tag_count = question.count("<image>")
                if tag_count == len(ref_full_paths):
                    for n in range(1, len(ref_full_paths) + 1):
                        question = question.replace(
                            "<image>", f"[reference image #{n}]", 1
                        )
                else:
                    print(
                        f"[Warning] MMSI sample {item.get('id', '?')}: "
                        f"{tag_count} <image> tags vs {len(ref_full_paths)} "
                        f"ref_images — leaving question verbatim"
                    )

            sample = MMSIVideoBenchSample(
                sample_id=item.get("id", ""),
                question=question,
                question_type=qtype,
                images=full_paths,
                answer=gt,
                image_groups=image_groups,
                frame_indices_groups=frame_indices_groups,
                fps_per_video=fps_per_video,
                total_frames_per_video=total_frames_per_video,
                duration_per_video=duration_per_video,
                video_names=video_names,
                choices=options,
                hint=item.get("hint", ""),
                video=video_path,
                fps=fps,
                duration_sec=duration_sec,
                total_video_frames=len(full_paths),
                num_videos=num_videos,
                ref_images=ref_full_paths,
            )
            self.data.append(sample)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        per_type: Dict[str, Dict[str, int]] = {}
        per_category: Dict[str, Dict[str, int]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer.strip().upper()
            is_correct = pred == gt
            if is_correct:
                correct += 1
            total += 1

            # Per-type
            qt = sample.question_type
            if qt not in per_type:
                per_type[qt] = {"correct": 0, "total": 0}
            per_type[qt]["total"] += 1
            if is_correct:
                per_type[qt]["correct"] += 1

            # Per-category
            cat = CATEGORY_MAP.get(qt, "Other")
            if cat not in per_category:
                per_category[cat] = {"correct": 0, "total": 0}
            per_category[cat]["total"] += 1
            if is_correct:
                per_category[cat]["correct"] += 1

            detailed.append({
                "id": sid, "type": qt, "ground_truth": gt,
                "prediction": pred_raw, "extracted": pred, "correct": is_correct,
            })

        results = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "per_type": {
                k: {**v, "accuracy": v["correct"] / max(v["total"], 1)}
                for k, v in per_type.items()
            },
            "per_category": {
                k: {**v, "accuracy": v["correct"] / max(v["total"], 1)}
                for k, v in per_category.items()
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results
