"""Output types for GPU model results.

These dataclasses are the single source of truth for all structured output
types returned by GPU models.  Both the GPU server and the agent kernel
import from here.

Design constraint: this file must import instantly on CPU-only nodes.
No torch dependency — all array fields are numpy.
"""

from dataclasses import dataclass, is_dataclass, fields
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


AGENT_CONTEXT_REGISTRY = set()


class AgentContext:
    """Base class for all structured data objects in the agent pipeline."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AGENT_CONTEXT_REGISTRY.add(cls)

    def to_message_content(self) -> str:
        return str(self)

    def get_computation_doc(self) -> Optional[Dict[str, str]]:
        return None

    def _get_obj_size_bytes(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, Image.Image):
            return obj.height * obj.width * len(obj.getbands())
        if is_dataclass(obj):
            return sum(self._get_obj_size_bytes(getattr(obj, f.name)) for f in fields(obj))
        if isinstance(obj, list):
            return sum(self._get_obj_size_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(self._get_obj_size_bytes(v) for v in obj.values())
        return 0

    def estimate_payload_size_mb(self):
        return self._get_obj_size_bytes(self) / (1024 ** 2)


@dataclass
class AgentToolOutput:
    result: Optional[Any] = None
    err_msg: Optional[str] = None
    err_src: Optional[str] = None

    @property
    def err(self) -> Optional[Dict[str, str]]:
        if self.err_msg:
            return {"msg": self.err_msg, "src": self.err_src}
        return None


# ---------------------------------------------------------------------------
# Reconstruction outputs (all arrays are numpy on the agent side)
# ---------------------------------------------------------------------------

@dataclass
class Pi3ReconstructionOutput(AgentContext):
    """Pi3 3D/4D reconstruction output. All arrays are numpy."""
    points: Any = None          # np.ndarray (N, H, W, 3)
    camera_poses: Any = None    # np.ndarray (N, 4, 4)
    confidence: Any = None      # np.ndarray (N, H, W)
    frame_indices: List[int] = None
    metric_scale: float = 1.0
    num_frames: int = 0
    _local_points: Any = None   # np.ndarray (N, H, W, 3)
    _rays: Any = None           # np.ndarray (N, H, W, 3)

    def to_message_content(self) -> str:
        return f"Pi3 reconstruction: {self.num_frames} frames"

    def get_computation_doc(self):
        return set(["extrinsic", "rotation", "homo_coord", "trajectory_transform"])


@dataclass
class DA3ReconstructionOutput(AgentContext):
    """DA3 reconstruction output. All arrays are numpy."""
    points: Any = None          # np.ndarray (N, H, W, 3)
    camera_poses: Any = None    # np.ndarray (N, 4, 4)
    confidence: Any = None      # np.ndarray (N, H, W)
    frame_indices: List[int] = None
    metric_scale: float = 1.0
    num_frames: int = 0
    _rays: Any = None
    _intrinsics: Any = None     # np.ndarray (N, 3, 3)

    def to_message_content(self) -> str:
        return f"DA3 reconstruction: {self.num_frames} frames"

    def get_computation_doc(self):
        return set(["extrinsic", "rotation", "homo_coord", "trajectory_transform"])


@dataclass
class MapAnythingReconstructionOutput(AgentContext):
    """MapAnything reconstruction output. All arrays are numpy."""
    points: Any = None          # np.ndarray (N, H, W, 3)
    camera_poses: Any = None    # np.ndarray (N, 4, 4)
    confidence: Any = None      # np.ndarray (N, H, W)
    frame_indices: List[int] = None
    metric_scale: float = 1.0
    num_frames: int = 0
    _rays: Any = None
    _intrinsics: Any = None     # np.ndarray (N, 3, 3)

    def to_message_content(self) -> str:
        return f"MapAnything reconstruction: {self.num_frames} frames"

    def get_computation_doc(self):
        return set(["extrinsic", "rotation", "homo_coord", "trajectory_transform"])


# ---------------------------------------------------------------------------
# Trajectory / projection outputs (all arrays are numpy)
# ---------------------------------------------------------------------------

@dataclass
class Pi3TrajectoryOutput(AgentContext):
    """Per-object 3D centroid trajectories. All arrays are numpy."""
    centroids_3d: Any = None    # np.ndarray (T, N_obj, 3)
    validity: Any = None        # np.ndarray (T, N_obj) bool
    labels: List[str] = None
    num_frames: int = 0
    num_objects: int = 0

    def to_message_content(self) -> str:
        if self.num_objects == 0:
            return "Trajectory: 0 objects tracked."
        summaries = []
        for i, label in enumerate(self.labels):
            valid_count = int(self.validity[:, i].sum())
            summaries.append(f"{label}: {valid_count}/{self.num_frames} valid frames")
        return f"3D trajectory for {self.num_objects} object(s): " + "; ".join(summaries)

    def get_computation_doc(self):
        return set(["trajectory_transform", "trajectory_velocity", "trajectory_relative"])


@dataclass
class Pi3ProjectionOutput(AgentContext):
    """Per-object 2D projections onto a camera frame. All arrays are numpy."""
    tracks_2d: Any = None       # np.ndarray (T, N_obj, 2)
    validity: Any = None        # np.ndarray (T, N_obj) bool
    labels: List[str] = None
    frame_idx: int = 0

    def to_message_content(self) -> str:
        n_obj = len(self.labels) if self.labels else 0
        return f"2D projection onto camera frame {self.frame_idx}: {n_obj} object(s)"

    def get_computation_doc(self):
        return set(["trajectory_transform"])


# ---------------------------------------------------------------------------
# SAM3 outputs (all arrays are numpy)
# ---------------------------------------------------------------------------

@dataclass
class SAM3ImageDetectionOutput(AgentContext):
    """SAM3 image detection output. All arrays are numpy."""
    boxes: Any = None           # np.ndarray (N, 4)
    scores: Any = None          # np.ndarray (N,)
    masks: Any = None           # np.ndarray (N, H, W) bool
    labels: List[str] = None

    def to_message_content(self) -> str:
        n = self.boxes.shape[0] if self.boxes is not None else 0
        return f"SAM3 detected {n} object(s)"


@dataclass
class SAM3VideoSegmentationOutput(AgentContext):
    """SAM3 video segmentation output. All arrays are numpy."""
    masks: Any = None           # np.ndarray (T, N_obj, H, W) bool
    object_ids: List[int] = None
    labels: List[str] = None
    frame_indices: List[int] = None
    num_frames: int = 0
    _per_frame_scores: Any = None  # np.ndarray (T, N_obj)

    def to_message_content(self) -> str:
        n_obj = len(self.object_ids) if self.object_ids else 0
        return f"SAM3 tracked {n_obj} object(s) across {self.num_frames} frames"
