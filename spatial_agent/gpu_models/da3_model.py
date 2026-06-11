"""DA3 (Depth Anything 3) 3D reconstruction.

Uses Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1) for monocular depth + camera pose
estimation, then unprojects depth to world-space point maps.  The output format
matches Pi3Model's ``reconstruct`` interface so the spatial agent can swap
backends transparently.
"""

import asyncio
import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image
import torch

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, gpu_inference_lock
from spatial_agent.gpu_models.types import DA3ReconstructionOutput  # noqa: F811
ImageLoader = None  # stub: not used (PIL images passed directly)

__all__ = ["DA3Model", "DA3ReconstructionOutput"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DA3_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "Depth-Anything-3", "src")
if _DA3_PATH not in sys.path:
    sys.path.insert(0, _DA3_PATH)

# DA3ReconstructionOutput is imported from types.py (lightweight, no torch dep)


def _preprocess_frames_da3(images: List[Image.Image]) -> tuple:
    """Crop+resize each PIL image to a single DA3 target shape.

    Picks the target shape from the *first* frame's aspect (mirrors Pi3's
    ``_preprocess_frames``) so all frames in the batch share (H, W) — this
    matches DA3's ``InputProcessor`` invariant (otherwise it would
    center-crop to the smallest size in the batch).

    Returns:
        (resized_images, long_edge) where ``long_edge = max(target_w, target_h)``.
        Caller passes ``long_edge`` as ``process_res`` so DA3's internal
        resize is a no-op.
    """
    from spatial_agent.gpu_models.image_resize import (
        da3_pick_target_shape,
        crop_to_aspect,
    )

    first = images[0]
    if first.mode == "RGBA":
        bg = Image.new("RGBA", first.size, (255, 255, 255, 255))
        first = Image.alpha_composite(bg, first)
    first = first.convert("RGB")
    target_w, target_h = da3_pick_target_shape(first.width, first.height)
    target_aspect = target_w / target_h

    out: List[Image.Image] = []
    for img in images:
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        if img.height > 0 and abs(img.width / img.height - target_aspect) > 1e-3:
            img = crop_to_aspect(img, target_aspect)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        out.append(img)

    return out, max(target_w, target_h)


class DA3Model(AgentTool):
    """Depth Anything 3 reconstruction model."""

    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 20.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 2

    MODEL_ID = os.environ.get(
        "SPATIAL_AGENT_DA3_MODEL_ID",
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    LOCAL_FILES_ONLY = os.environ.get("SPATIAL_AGENT_DA3_LOCAL_FILES_ONLY", "1") != "0"
    DEVICE = os.environ.get("SPATIAL_AGENT_DA3_DEVICE", "cuda")

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()

        if _DA3_PATH not in sys.path:
            sys.path.insert(0, _DA3_PATH)

        from depth_anything_3.api import DepthAnything3

        self.model = (
            DepthAnything3.from_pretrained(
                self.MODEL_ID,
                local_files_only=self.LOCAL_FILES_ONLY,
            )
            .to(self.DEVICE)
            .eval()
        )
        self.image_loader = image_loader
        self.infer_lock = asyncio.Lock()

    @torch.no_grad()
    def _reconstruct(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> DA3ReconstructionOutput:
        """Run DA3 inference and unproject depth to world-space point maps."""
        N = len(images)
        fi = list(frame_indices) if frame_indices is not None else list(range(N))

        # Snap all frames to a single DA3 training shape (one of DA3_TARGET_SHAPES).
        # Passing process_res = long edge of that shape makes DA3's internal
        # InputProcessor a no-op (longest_side already matches, dims already
        # multiples of 14), preserving our LANCZOS resize.
        images, long_edge = _preprocess_frames_da3(images)

        # Run DA3 inference - returns Prediction with depth, extrinsics,
        # intrinsics, and optional confidence/sky/scale fields.
        device_type = torch.device(self.DEVICE).type
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type, dtype=dtype, enabled=device_type == "cuda"):
            prediction = self.model.inference(
                images,
                process_res=long_edge,
                process_res_method="upper_bound_resize",
            )

        # Extract numpy arrays from Prediction
        depth = prediction.depth.astype(np.float32)  # (N, H', W')
        conf = getattr(prediction, "conf", None)  # (N, H', W') or None
        if conf is not None:
            conf = conf.astype(np.float32)
        else:
            # If no confidence, use ones
            conf = np.ones_like(depth)

        # Zero out confidence for sky pixels so they are excluded from
        # downstream processing (BEV rendering, point cloud filtering, etc.)
        sky = getattr(prediction, "sky", None)
        if sky is not None:
            sky_mask = sky > 0.3  # True = sky
            conf[sky_mask] = 0.0

        # Extrinsics: DA3 returns w2c (N, 3, 4) or (N, 4, 4)
        ext_raw = prediction.extrinsics
        if ext_raw is None:
            raise RuntimeError(
                "DA3 model did not produce extrinsics. "
                "The DA3NESTED model is required for multi-view reconstruction."
            )
        if ext_raw.shape[-2] == 3:
            # Pad (N, 3, 4) → (N, 4, 4)
            pad = np.zeros((N, 1, 4), dtype=ext_raw.dtype)
            pad[:, 0, 3] = 1.0
            w2c = np.concatenate([ext_raw, pad], axis=1)  # (N, 4, 4)
        else:
            w2c = ext_raw.copy()

        # Convert w2c → c2w by inversion
        c2w = np.zeros_like(w2c, dtype=np.float64)
        for i in range(N):
            c2w[i] = np.linalg.inv(w2c[i].astype(np.float64))

        # Intrinsics: (N, 3, 3) — DA3NESTED always provides these
        if prediction.intrinsics is None:
            raise RuntimeError(
                "DA3 model did not produce intrinsics. "
                "The DA3NESTED model is required for multi-view reconstruction."
            )
        intrinsics = prediction.intrinsics.astype(np.float64)  # (N, 3, 3)

        # Unproject depth to world-space point maps
        H, W = depth.shape[1], depth.shape[2]
        # Build pixel grid
        u_coords = np.arange(W, dtype=np.float64)
        v_coords = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W)

        points = np.zeros((N, H, W, 3), dtype=np.float32)
        for i in range(N):
            fx = intrinsics[i, 0, 0]
            fy = intrinsics[i, 1, 1]
            cx = intrinsics[i, 0, 2]
            cy = intrinsics[i, 1, 2]
            d = depth[i].astype(np.float64)  # (H, W)

            # Camera-space coordinates
            x_cam = (uu - cx) * d / fx
            y_cam = (vv - cy) * d / fy
            z_cam = d

            # Stack to (H*W, 3) and transform to world
            cam_pts = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)  # (H*W, 3)
            ones = np.ones((cam_pts.shape[0], 1), dtype=np.float64)
            homo = np.hstack([cam_pts, ones])  # (H*W, 4)
            world_pts = (c2w[i] @ homo.T).T[:, :3]  # (H*W, 3)
            points[i] = world_pts.reshape(H, W, 3).astype(np.float32)

        # Metric scale from DA3. The nested model depth is already in meters;
        # scale_factor is kept when the API exposes it.
        scale_factor = getattr(prediction, "scale_factor", None)
        metric_scale = float(scale_factor) if scale_factor is not None else 1.0

        return DA3ReconstructionOutput(
            points=points,
            camera_poses=c2w,
            confidence=conf,
            frame_indices=fi,
            metric_scale=metric_scale,
            num_frames=N,
            _rays=None,
            _intrinsics=intrinsics,
        )

    def _reconstruct_locked(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> DA3ReconstructionOutput:
        """Acquire the per-process GPU FIFO lock, then run reconstruction."""
        with gpu_inference_lock():
            return self._reconstruct(images, frame_indices)

    @AgentTool.document_output_class(DA3ReconstructionOutput)
    async def reconstruct(
        self,
        video_frames: List[Image.Image],
        frame_indices: Optional[List[int]] = None,
    ) -> AgentToolOutput:
        """Perform depth estimation and 3D reconstruction using Depth Anything 3.

        Args:
            video_frames: List of PIL images.
            frame_indices: Absolute video frame indices.
        """
        if not isinstance(video_frames, list):
            video_frames = [video_frames]

        async with self.infer_lock:
            output = await asyncio.to_thread(self._reconstruct_locked, video_frames, frame_indices)

        return self.success(result=output)
