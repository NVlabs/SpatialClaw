import asyncio
from dataclasses import dataclass
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as TF

from spatial_agent.gpu_models.types import (
    Pi3ReconstructionOutput,
    Pi3TrajectoryOutput,
    Pi3ProjectionOutput,
)

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, AgentContext, gpu_inference_lock
ImageLoader = None  # stub: not used (PIL images passed directly)

__all__ = [
    'Pi3Model',
    'Pi3ReconstructionOutput',
    'Pi3TrajectoryOutput',
    'Pi3ProjectionOutput',
]

# Ensure Pi3 third-party path is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_PI3_PATH = os.path.join(_PROJECT_ROOT, 'tools', 'third_party', 'Pi3')
if _PI3_PATH not in sys.path:
    sys.path.insert(0, _PI3_PATH)


def _preprocess_frames(images: List[Image.Image]) -> torch.Tensor:
    """Convert PIL images to a (1, N, 3, H, W) tensor in [0, 1].

    Uses Pi3's official training-distribution sizing: pixel area approximately
    PI3_PIXEL_LIMIT (= 255_000) with H, W as multiples of 14, and no padding.
    All frames are projected onto the same (k*14, m*14) grid via center-crop
    to Pi3's preferred aspect followed by LANCZOS resize.
    """
    from spatial_agent.gpu_models.image_resize import (
        crop_to_aspect,
        pi3_training_grid,
        PI3_PATCH_SIZE,
    )

    to_tensor = TF.ToTensor()

    # Pick (k, m) from the first frame's aspect; all frames are expected to
    # share that aspect after upstream InputImages preprocessing.
    first = images[0]
    if first.mode == 'RGBA':
        background = Image.new('RGBA', first.size, (255, 255, 255, 255))
        first = Image.alpha_composite(background, first)
    first = first.convert('RGB')
    k, m = pi3_training_grid(first.width, first.height)
    target_w, target_h = k * PI3_PATCH_SIZE, m * PI3_PATCH_SIZE
    target_aspect = k / m

    tensors = []
    for img in images:
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
        if img.height > 0 and abs(img.width / img.height - target_aspect) > 1e-3:
            img = crop_to_aspect(img, target_aspect)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        tensors.append(to_tensor(img))

    imgs = torch.stack(tensors)  # (N, 3, H, W)
    return imgs.unsqueeze(0)  # (1, N, 3, H, W)


# Output types (Pi3ReconstructionOutput, Pi3TrajectoryOutput, Pi3ProjectionOutput)
# are imported from types.py — single source of truth, no torch dependency.


class Pi3Model(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 15.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 2

    MODEL_ID = 'yyfz233/Pi3X'
    DEVICE = 'cuda'

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()

        if _PI3_PATH not in sys.path:
            sys.path.insert(0, _PI3_PATH)

        from pi3.models.pi3x import Pi3X
        self.model = Pi3X.from_pretrained(self.MODEL_ID).to(self.DEVICE).eval()
        self.image_loader = image_loader
        self.infer_lock = asyncio.Lock()

    @torch.no_grad()
    def _reconstruct(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> Pi3ReconstructionOutput:
        imgs = _preprocess_frames(images).to(self.DEVICE)  # (1, N, 3, H, W)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast('cuda', dtype=dtype):
            res = self.model(imgs=imgs)

        # res keys: points, local_points, rays, conf, camera_poses, metric
        # Squeeze batch dim, move to CPU, convert to numpy (agent is GPU-unaware)
        points = res['points'][0].cpu().detach().numpy()          # (N, H, W, 3)
        conf_raw = res['conf'][0].cpu().detach()                  # (N, H, W, 1)
        confidence = torch.sigmoid(conf_raw[..., 0]).numpy()      # (N, H, W)
        camera_poses = res['camera_poses'][0].cpu().detach().numpy()  # (N, 4, 4)
        metric = res['metric'][0].cpu().item()
        local_points = res['local_points'][0].cpu().detach().numpy()  # (N, H, W, 3)
        rays = res['rays'][0].cpu().detach().numpy()               # (N, H, W, 3)

        N = points.shape[0]
        fi = list(frame_indices) if frame_indices is not None else list(range(N))

        del res, imgs
        torch.cuda.empty_cache()

        return Pi3ReconstructionOutput(
            points=points,
            camera_poses=camera_poses,
            confidence=confidence,
            frame_indices=fi,
            metric_scale=float(metric),
            num_frames=N,
            _local_points=local_points,
            _rays=rays,
        )

    def _reconstruct_locked(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> Pi3ReconstructionOutput:
        """Acquire per-GPU file lock, then run reconstruction."""
        with gpu_inference_lock():
            return self._reconstruct(images, frame_indices)

    def _extract_object_trajectory(
        self,
        reconstruction: Pi3ReconstructionOutput,
        video_segmentation,  # SAM3VideoSegmentationOutput
        conf_threshold: float = 0.1,
        min_pixels: int = 10,
    ) -> Pi3TrajectoryOutput:
        """CPU-side trajectory extraction: align SAM3 masks with Pi3X points by frame_indices."""
        from spatial_agent.gpu_models.sam3_model import SAM3VideoSegmentationOutput

        T_recon = reconstruction.num_frames
        n_obj = video_segmentation.num_objects if hasattr(video_segmentation, 'num_objects') else len(video_segmentation.object_ids)
        labels = video_segmentation.labels

        centroids = np.zeros((T_recon, n_obj, 3), dtype=np.float32)
        validity = np.zeros((T_recon, n_obj), dtype=bool)

        # Build a lookup from absolute frame index → segmentation frame index
        seg_frame_to_t = {abs_idx: t for t, abs_idx in enumerate(video_segmentation.frame_indices)}

        for recon_t, abs_idx in enumerate(reconstruction.frame_indices):
            seg_t = seg_frame_to_t.get(abs_idx, None)
            if seg_t is None:
                continue

            pts = np.asarray(reconstruction.points[recon_t])       # (H, W, 3)
            conf = np.asarray(reconstruction.confidence[recon_t])  # (H, W)

            for n in range(n_obj):
                mask = np.asarray(video_segmentation.masks[seg_t, n], dtype=bool)  # (H, W)
                # Resize mask to match reconstruction spatial dims if needed
                if mask.shape != pts.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_h = pts.shape[0] / mask.shape[0]
                    scale_w = pts.shape[1] / mask.shape[1]
                    mask = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5

                valid_mask = mask & (conf > conf_threshold)
                if valid_mask.sum() >= min_pixels:
                    masked_pts = pts[valid_mask]  # (K, 3)
                    centroids[recon_t, n] = masked_pts.mean(axis=0)
                    validity[recon_t, n] = True

        return Pi3TrajectoryOutput(
            centroids_3d=centroids,
            validity=validity,
            labels=labels,
            num_frames=T_recon,
            num_objects=n_obj,
        )

    def _project_to_camera_frame(
        self,
        reconstruction: Pi3ReconstructionOutput,
        trajectory: Pi3TrajectoryOutput,
        frame_idx: int = 0,
    ) -> Pi3ProjectionOutput:
        """Project 3D world trajectories onto a target camera's image plane."""
        T = trajectory.num_frames
        n_obj = trajectory.num_objects

        c2w = np.asarray(reconstruction.camera_poses[frame_idx], dtype=np.float64)  # (4, 4)
        w2c = np.linalg.inv(c2w)  # (4, 4)

        # Estimate intrinsics from rays
        rays = np.asarray(reconstruction._rays[frame_idx], dtype=np.float64)  # (H, W, 3)
        H, W = rays.shape[:2]

        cy, cx = H / 2.0, W / 2.0
        ray_right = rays[H // 2, W - 1]  # (3,)
        ray_bottom = rays[H - 1, W // 2]  # (3,)
        fx = abs((W / 2.0) / (ray_right[0] / ray_right[2] + 1e-8))
        fy = abs((H / 2.0) / (ray_bottom[1] / ray_bottom[2] + 1e-8))

        tracks_2d = np.zeros((T, n_obj, 2), dtype=np.float32)
        proj_validity = np.array(trajectory.validity, dtype=bool).copy()

        for t in range(T):
            for n in range(n_obj):
                if not trajectory.validity[t, n]:
                    continue
                p_world = np.asarray(trajectory.centroids_3d[t, n], dtype=np.float64)
                p_world_h = np.append(p_world, 1.0)  # (4,)
                p_cam = w2c @ p_world_h  # (4,)
                if p_cam[2] <= 0:
                    proj_validity[t, n] = False
                    continue
                tracks_2d[t, n, 0] = float(fx * p_cam[0] / p_cam[2] + cx)
                tracks_2d[t, n, 1] = float(fy * p_cam[1] / p_cam[2] + cy)

        return Pi3ProjectionOutput(
            tracks_2d=tracks_2d,
            validity=proj_validity,
            labels=trajectory.labels,
            frame_idx=frame_idx,
        )

    @AgentTool.document_output_class(Pi3ReconstructionOutput)
    async def reconstruct(
        self,
        video_frames: List[Image.Image],
        frame_indices: Optional[List[int]] = None,
    ) -> AgentToolOutput:
        """
        Performs feed-forward 4D reconstruction from a sequence of video frames. Outputs per-frame dense 3D point clouds, camera poses (camera-to-world), and reconstruction confidence, all in a unified world coordinate system.
        Args:
            video_frames (List[Image.Image]): VLM-selected subset of video frames. Default: 8 uniformly sampled frames from `$input_images.images` (e.g., `$input_images.images[::4]`).
            frame_indices (Optional[List[int]]): Absolute video frame indices corresponding to each frame in `video_frames`. Pass `$input_images.frame_indices[::4]` (sliced to match `video_frames`). Stored in the output for downstream alignment with SAM3 masks.
        """
        if not isinstance(video_frames, list):
            video_frames = [video_frames]

        async with self.infer_lock:
            output = await asyncio.to_thread(self._reconstruct_locked, video_frames, frame_indices)

        return self.success(result=output)

    @AgentTool.document_output_class(Pi3TrajectoryOutput)
    async def extract_object_trajectory(
        self,
        reconstruction: Pi3ReconstructionOutput,
        video_segmentation,
        conf_threshold: float = 0.1,
        min_pixels: int = 10,
    ) -> AgentToolOutput:
        """
        Extracts per-object 3D centroid trajectories by combining Pi3X reconstruction geometry with SAM3 segmentation masks. For each frame and object, computes the mean 3D position of the masked and confident 3D points.
        Args:
            reconstruction (Pi3ReconstructionOutput): 3D reconstruction from `Pi3Model.reconstruct`.
            video_segmentation (SAM3VideoSegmentationOutput): Video segmentation from `SAM3Model.segment_video`. Frame alignment is handled automatically via `frame_indices`.
            conf_threshold (float): Minimum reconstruction confidence to include a point. Default: 0.1.
            min_pixels (int): Minimum number of valid masked pixels required for a valid centroid. Default: 10.
        """
        output = await asyncio.to_thread(
            self._extract_object_trajectory,
            reconstruction,
            video_segmentation,
            conf_threshold,
            min_pixels,
        )
        return self.success(result=output)

    @AgentTool.document_output_class(Pi3ProjectionOutput)
    async def project_to_camera_frame(
        self,
        reconstruction: Pi3ReconstructionOutput,
        trajectory: Pi3TrajectoryOutput,
        frame_idx: int = 0,
    ) -> AgentToolOutput:
        """
        Projects 3D world-space trajectories onto a specific camera frame's image plane to obtain 2D pixel coordinates. Useful for analyzing motion direction from the camera's perspective.
        Args:
            reconstruction (Pi3ReconstructionOutput): Reconstruction from `Pi3Model.reconstruct` (provides camera poses and intrinsics via ray directions).
            trajectory (Pi3TrajectoryOutput): 3D trajectory from `Pi3Model.extract_object_trajectory`.
            frame_idx (int): Index into the reconstruction's frame list (0-based) for the target camera. Default: 0 (first reconstructed frame = camera 0's viewpoint).
        """
        output = await asyncio.to_thread(
            self._project_to_camera_frame,
            reconstruction,
            trajectory,
            frame_idx,
        )
        return self.success(result=output)
