"""MapAnything 3D reconstruction backend.

Uses Meta's MapAnything feed-forward metric reconstruction model and adapts its
OpenCV camera-to-world outputs to the same shape consumed by ReconstructTool.
"""

import asyncio
import os
import sys
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image
import torch

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, gpu_inference_lock
from spatial_agent.gpu_models.types import MapAnythingReconstructionOutput

ImageLoader = None  # stub: not used (PIL images passed directly)

__all__ = ["MapAnythingModel", "MapAnythingReconstructionOutput"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MAPANYTHING_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "map-anything")
_TORCH_HUB_DIR = os.environ.get(
    "SPATIAL_AGENT_TORCH_HUB_DIR",
    os.path.join(_PROJECT_ROOT, "tools", "third_party", "torch_hub"),
)
if _MAPANYTHING_PATH not in sys.path:
    sys.path.insert(0, _MAPANYTHING_PATH)


class MapAnythingModel(AgentTool):
    """MapAnything metric reconstruction model."""

    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 24.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 2

    MODEL_ID = os.environ.get(
        "SPATIAL_AGENT_MAPANYTHING_MODEL_ID",
        "facebook/map-anything",
    )
    LOCAL_FILES_ONLY = os.environ.get("SPATIAL_AGENT_MAPANYTHING_LOCAL_FILES_ONLY", "1") != "0"
    DEVICE = os.environ.get("SPATIAL_AGENT_MAPANYTHING_DEVICE", "cuda")
    RESOLUTION_SET = int(os.environ.get("SPATIAL_AGENT_MAPANYTHING_RESOLUTION_SET", "518"))

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()

        if _MAPANYTHING_PATH not in sys.path:
            sys.path.insert(0, _MAPANYTHING_PATH)

        os.makedirs(_TORCH_HUB_DIR, exist_ok=True)
        torch.hub.set_dir(_TORCH_HUB_DIR)

        from mapanything.models import MapAnything

        try:
            self.model = (
                MapAnything.from_pretrained(
                    self.MODEL_ID,
                    local_files_only=self.LOCAL_FILES_ONLY,
                )
                .to(self.DEVICE)
                .eval()
            )
        except Exception as exc:
            if self.LOCAL_FILES_ONLY:
                raise RuntimeError(
                    "MapAnything weights or DINOv2 torch-hub encoder are not available "
                    "in the local caches. Pre-download facebook/map-anything and "
                    "facebookresearch/dinov2:dinov2_vitg14, or set "
                    "SPATIAL_AGENT_MAPANYTHING_LOCAL_FILES_ONLY=0 for first launch."
                ) from exc
            raise

        self.image_loader = image_loader
        self.infer_lock = asyncio.Lock()

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _squeeze_view_array(value, name: str, expected_last_dim: Optional[int] = None) -> np.ndarray:
        arr = MapAnythingModel._to_numpy(value)
        if arr is None:
            raise RuntimeError(f"MapAnything prediction is missing {name!r}.")

        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        if expected_last_dim is not None:
            if arr.ndim < 1 or arr.shape[-1] != expected_last_dim:
                raise RuntimeError(
                    f"MapAnything {name!r} has shape {arr.shape}, expected last dim {expected_last_dim}."
                )
        return arr

    @staticmethod
    def _extract_map(value, name: str, target_shape) -> np.ndarray:
        arr = MapAnythingModel._to_numpy(value)
        if arr is None:
            raise RuntimeError(f"MapAnything prediction is missing {name!r}.")

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape != target_shape:
            raise RuntimeError(
                f"MapAnything {name!r} has shape {arr.shape}, expected {target_shape}."
            )
        return arr

    def _preprocess_views(self, images: List[Image.Image]):
        from mapanything.utils.image import preprocess_inputs

        views = [{"img": image.convert("RGB")} for image in images]
        return preprocess_inputs(
            views,
            resize_mode="fixed_mapping",
            norm_type="dinov2",
            patch_size=14,
            resolution_set=self.RESOLUTION_SET,
            verbose=False,
        )

    @torch.no_grad()
    def _reconstruct(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> MapAnythingReconstructionOutput:
        """Run MapAnything inference and return world-space point maps."""
        N = len(images)
        fi = list(frame_indices) if frame_indices is not None else list(range(N))

        processed_views = self._preprocess_views(images)
        predictions = self.model.infer(
            processed_views,
            memory_efficient_inference=True,
            minibatch_size=None,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            # mask_edges=True was zeroing out edge confidence, which left
            # small/thin object masks with too few valid points for a stable
            # 3D centroid (sample-109 style backward/right BEV artifact).
            # Disable so edges retain their natural (low) confidence.
            mask_edges=False,
            apply_confidence_mask=False,
            use_multiview_confidence=False,
        )
        if len(predictions) != N:
            raise RuntimeError(
                f"MapAnything returned {len(predictions)} predictions for {N} input frames."
            )

        points_list = []
        poses_list = []
        confidence_list = []
        intrinsics_list = []
        rays_list = []
        metric_scales = []

        for pred in predictions:
            points = self._squeeze_view_array(pred.get("pts3d"), "pts3d", expected_last_dim=3)
            if points.ndim != 3:
                raise RuntimeError(f"MapAnything 'pts3d' has shape {points.shape}, expected (H, W, 3).")
            H, W = points.shape[:2]

            pose = self._squeeze_view_array(pred.get("camera_poses"), "camera_poses")
            if pose.shape != (4, 4):
                raise RuntimeError(
                    f"MapAnything 'camera_poses' has shape {pose.shape}, expected (4, 4)."
                )

            intrinsics = self._squeeze_view_array(pred.get("intrinsics"), "intrinsics")
            if intrinsics.shape != (3, 3):
                raise RuntimeError(
                    f"MapAnything 'intrinsics' has shape {intrinsics.shape}, expected (3, 3)."
                )

            if pred.get("conf") is not None:
                confidence = self._extract_map(pred.get("conf"), "conf", (H, W)).astype(np.float32)
            elif pred.get("mask") is not None:
                confidence = self._extract_map(pred.get("mask"), "mask", (H, W)).astype(np.float32)
            else:
                confidence = np.ones((H, W), dtype=np.float32)

            if pred.get("mask") is not None:
                mask = self._extract_map(pred.get("mask"), "mask", (H, W)).astype(np.float32)
                confidence = confidence * mask

            if pred.get("ray_directions") is not None:
                rays = self._squeeze_view_array(
                    pred.get("ray_directions"),
                    "ray_directions",
                    expected_last_dim=3,
                ).astype(np.float32)
            else:
                rays = np.zeros((H, W, 3), dtype=np.float32)

            scale = pred.get("metric_scaling_factor")
            if scale is not None:
                scale_arr = self._to_numpy(scale).reshape(-1)
                if scale_arr.size:
                    metric_scales.append(float(scale_arr[0]))

            points_list.append(points.astype(np.float32))
            poses_list.append(pose.astype(np.float64))
            confidence_list.append(confidence.astype(np.float32))
            intrinsics_list.append(intrinsics.astype(np.float64))
            rays_list.append(rays)

        points_np = np.stack(points_list, axis=0)
        camera_poses_np = np.stack(poses_list, axis=0)
        confidence_np = np.stack(confidence_list, axis=0)
        intrinsics_np = np.stack(intrinsics_list, axis=0)
        rays_np = np.stack(rays_list, axis=0)
        metric_scale = float(np.mean(metric_scales)) if metric_scales else 1.0

        del predictions, processed_views
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return MapAnythingReconstructionOutput(
            points=points_np,
            camera_poses=camera_poses_np,
            confidence=confidence_np,
            frame_indices=fi,
            metric_scale=metric_scale,
            num_frames=N,
            _rays=rays_np,
            _intrinsics=intrinsics_np,
        )

    def _reconstruct_locked(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> MapAnythingReconstructionOutput:
        """Acquire the per-process GPU FIFO lock, then run reconstruction."""
        with gpu_inference_lock():
            return self._reconstruct(images, frame_indices)

    @AgentTool.document_output_class(MapAnythingReconstructionOutput)
    async def reconstruct(
        self,
        video_frames: List[Image.Image],
        frame_indices: Optional[List[int]] = None,
    ) -> AgentToolOutput:
        """Perform metric 3D reconstruction using MapAnything.

        Args:
            video_frames: List of PIL images.
            frame_indices: Absolute video frame indices.
        """
        if not isinstance(video_frames, list):
            video_frames = [video_frames]

        async with self.infer_lock:
            output = await asyncio.to_thread(self._reconstruct_locked, video_frames, frame_indices)

        return self.success(result=output)
