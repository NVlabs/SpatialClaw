"""Reconstruct GPU tool: 4D reconstruction with gravity alignment.

Wraps Pi3Model, DA3Model, or MapAnythingModel on the GPU server and converts outputs to
spatial_agent per-frame numpy types.  The backend is selected by config.
"""

from collections import OrderedDict
from typing import List, Optional

import numpy as np
from PIL import Image

from spatial_agent.kernel_types.per_frame_types import (
    PerFrameDepth,
    PerFrameExtrinsics,
    PerFrameIntrinsics,
    PerFramePointMap,
    Reconstruction,
)
from spatial_agent.tools.base import GPUTool, ensure_image_list
from spatial_agent.tools.geometry_utils import GeometryUtils


class ReconstructTool(GPUTool):
    """Client wrapper for the Reconstruct GPU server.

    Usage::

        recon = tools.Reconstruct.Reconstruct(InputImages[:32])
        bev = recon.render_bev(masks=seg)  # VisualFeedback

    """

    TOOL_ABLATION_PREFIX = "tool_reconstruct"

    TOOL_PROMPT_SECTIONS = OrderedDict([
        ("api", """\
### tools.Reconstruct - 3D/4D Reconstruction (GPU)
- `tools.Reconstruct.Reconstruct(frames)` → `Reconstruction`
  - **Max {reconstruct_max_frames} frames.** Raises error if exceeded. Subsample first.
  - Best quality with 8-32 frames; diminishing returns beyond that.
  - Frame indices are auto-extracted from `InputImages`. Pass `frame_indices=` only if using raw PIL images.
  - Output is gravity-aligned: +Y = up in world frame, ground plane at Y ≈ 0. First camera looks toward -Z.
  - Output spatial resolution matches the input frame resolution — no resizing needed when combining with SAM3 masks.

**Reconstruction attributes** (exact access patterns — use absolute frame indices):
```
recon.frame_indices          # list[int] — absolute frame indices
recon.num_frames             # int — number of reconstructed frames
recon.metric_scale           # float — scale factor (points in meters)

# Access by ABSOLUTE frame index (NOT local array index):
recon.depth[frame_idx]       # (H, W) float32 — depth at frame
recon.extrinsics[frame_idx]  # (4, 4) float64 — camera-to-world SE(3)
recon.intrinsics[frame_idx]  # dict(fx=, fy=, cx=, cy=) — intrinsics at frame
recon.points[frame_idx]      # (H, W, 3) float32 — XYZ world coords at frame

# Raw batch arrays (use absolute-index API above when possible):
recon.points.points          # np.ndarray (N, H, W, 3) float32
recon.points.confidence      # np.ndarray (N, H, W)   float32
recon.extrinsics.camera_poses  # np.ndarray (N, 4, 4) float64
```
"""),
        ("bev", """\
**BEV rendering** — `recon.render_bev(...)` → `VisualFeedback`
Use BEV when you need to understand spatial layout, relative positions, or movement patterns of objects from a top-down perspective — e.g., "is the car to the left or right of the person?", "which direction is the dog moving?", or "how far apart are the two buildings?"

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `masks` | `PerFrameMask` or `(N, N_obj, H, W)` bool | `None` | Pre-computed segmentation masks |
| `labels` | `list[str]` | `None` | Object labels (required when masks is a raw array) |
| `ref_frame` | `int` | `0` | Absolute frame index (from `recon.frame_indices`). Default 0 uses the first reconstructed frame. |
| `ego_trajectory` | `bool` | `True` | Draw camera egomotion path (green→yellow dashed) |

Renders a top-down BEV with object annotations and optional camera egomotion trajectory.
**Requires masks or ego_trajectory** — without either, the plot is empty (no pointcloud is drawn).
Objects are annotated with automatic motion detection:
- **Stationary objects** → labelled oriented bounding box
- **Moving objects** → colour-graded trajectory line with arrowhead

**Reading trajectories**: The line colour goes from **red** (first frame) to **blue** (last frame), showing temporal progression. The **arrowhead** at the end points in the direction of motion. A **red dot** marks the starting position. The label is placed at the trajectory midpoint.

Works with any number of frames (1 or more). With a single frame, all objects get bounding boxes (no trajectory possible).

```python
# Reconstruct from InputImages (frame indices auto-extracted)
recon = tools.Reconstruct.Reconstruct(InputImages[:32])

# BEV with object annotations from SAM3 masks
recon.render_bev(masks=seg)
recon.render_bev(masks=seg, labels=seg.labels)

# BEV with egomotion trajectory (camera path)
recon.render_bev(masks=seg, ego_trajectory=True)

# BEV with raw mask array + explicit labels
recon.render_bev(masks=mask_arr, labels=["car", "person"])

# Re-render with different reference camera (use absolute frame index)
fi = recon.frame_indices[5]  # get absolute frame index
recon.render_bev(masks=seg, ref_frame=fi)
```
"""),
        ("pose_convention", """\
**Camera pose convention** (camera-to-world):
```
pose = recon.extrinsics[frame_idx]  # (4, 4) — use absolute frame index
cam_pos    = pose[:3, 3]   # camera position in world
cam_right  = pose[:3, 0]   # camera +X axis (right in image) in world
cam_down   = pose[:3, 1]   # camera +Y axis (down in image) in world — NOT world up!
cam_fwd    = pose[:3, 2]   # camera +Z axis (forward/into scene) in world
```
"""),
    ])

    # Legacy alias — preserved so any caller that reads
    # ``cls.TOOL_PROMPT_DESCRIPTION`` directly (instead of going through
    # ``get_prompt_description``) still gets the full text. The ablation-aware
    # path uses TOOL_PROMPT_SECTIONS above.
    TOOL_PROMPT_DESCRIPTION = "\n".join(TOOL_PROMPT_SECTIONS.values())

    def __init__(self, sam3_tool, config,
                 gpu_tool_max_retries: int = 3, metadata=None):
        super().__init__(deployment_name="spatial_Reconstruct",
                         gpu_tool_max_retries=gpu_tool_max_retries)
        self._sam3_tool = sam3_tool  # SAM3Tool instance (duck-typed)
        self.MAX_FRAMES = getattr(config, "reconstruct_max_frames", 32)
        # When metadata is absent (e.g. ad-hoc tests), default to is_video=True
        # to preserve the existing trajectory-line behaviour.
        self._is_video = bool(getattr(metadata, "is_video", True))

    def Reconstruct(
        self,
        frames: List[Image.Image],
        frame_indices: Optional[List[int]] = None,
    ) -> Reconstruction:
        """Perform 3D/4D reconstruction.

        Args:
            frames: List of PIL images or FrameImage objects (max ``MAX_FRAMES``).
                When FrameImage objects are passed, frame_indices are
                auto-extracted.
            frame_indices: Absolute video frame indices.  Optional — if
                ``None`` and *frames* contain ``FrameImage`` objects, indices
                are auto-extracted from them.

        Returns:
            ``Reconstruction`` with gravity-aligned point maps, depth,
            intrinsics, extrinsics, and a BEV ``VisualFeedback``.
        """
        frames = ensure_image_list(frames)
        if len(frames) == 0:
            raise ValueError(
                "Reconstruct requires at least 1 frame. "
                "Pass InputImages[:N] or a list of PIL images."
            )
        if len(frames) > self.MAX_FRAMES:
            n = len(frames)
            step = n // self.MAX_FRAMES + 1
            raise ValueError(
                f"Reconstruct accepts at most {self.MAX_FRAMES} frames, "
                f"got {n}. Subsample first, e.g.:\n"
                f"  recon = tools.Reconstruct.Reconstruct(InputImages[::{step}])"
            )

        # Auto-extract frame_indices from FrameImage objects
        from spatial_agent.kernel_types.frame_image import FrameImage

        if frame_indices is None and len(frames) > 0 and isinstance(frames[0], FrameImage):
            frame_indices = [f.frame_index for f in frames]
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        if len(frame_indices) != len(frames):
            raise ValueError(
                f"frame_indices length ({len(frame_indices)}) must match "
                f"frames length ({len(frames)})."
            )

        # Unwrap FrameImage → PIL for serialization
        plain_frames = [f.image if isinstance(f, FrameImage) else f for f in frames]

        # 1. Call reconstruction GPU server
        raw = self._call_remote(
            "reconstruct",
            video_frames=plain_frames,
            frame_indices=frame_indices,
        )
        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"Reconstruction failed: {raw.err['msg']}")
        raw_result = raw.result if hasattr(raw, "result") else raw

        # 2. Extract numpy arrays from raw result
        points_np = self._to_numpy(raw_result.points)  # (N, H_model, W_model, 3)
        confidence_np = self._to_numpy(raw_result.confidence)  # (N, H_model, W_model)
        camera_poses_np = self._to_numpy(raw_result.camera_poses)  # (N, 4, 4)
        metric_scale = float(raw_result.metric_scale)

        # 2b. Resize point maps and confidence to match input frame resolution
        H_target, W_target = plain_frames[0].height, plain_frames[0].width
        H_model, W_model = points_np.shape[1], points_np.shape[2]
        if (H_model, W_model) != (H_target, W_target):
            points_np = self._resize_spatial(points_np, H_target, W_target)
            confidence_np = self._resize_spatial(confidence_np, H_target, W_target)

        # Compute depth from points (z-coordinate in camera frame)
        N = points_np.shape[0]
        depth_np = np.zeros(confidence_np.shape, dtype=np.float32)
        for i in range(N):
            w2c = np.linalg.inv(camera_poses_np[i])
            flat_pts = points_np[i].reshape(-1, 3)
            ones = np.ones((flat_pts.shape[0], 1), dtype=np.float32)
            homo = np.hstack([flat_pts, ones])
            cam_pts = (w2c @ homo.T).T[:, 2]
            depth_np[i] = cam_pts.reshape(points_np.shape[1], points_np.shape[2])

        # 3. Ground plane detection + gravity alignment
        gravity_dir, rotation, ground_mask = self._find_gravity(
            points_np, confidence_np, camera_poses_np, plain_frames
        )

        if rotation is not None:
            # Apply rotation to point maps
            R4 = np.eye(4, dtype=np.float64)
            R4[:3, :3] = rotation
            for i in range(N):
                flat = points_np[i].reshape(-1, 3)
                points_np[i] = (rotation @ flat.T).T.reshape(points_np[i].shape)
            # Apply rotation to extrinsics
            for i in range(N):
                camera_poses_np[i] = R4 @ camera_poses_np[i]

        # 3b. Translate so ground plane is at Y=0
        ground_y = self._estimate_ground_y(
            points_np, confidence_np, ground_mask
        )
        if ground_y is not None:
            for i in range(N):
                points_np[i][..., 1] -= ground_y
            camera_poses_np[:, 1, 3] -= ground_y

        # 3c. Align first camera forward direction to -Z (yaw rotation around Y)
        self._align_forward_to_neg_z(points_np, camera_poses_np)

        # 4. Compute intrinsics at the target (input frame) resolution
        fx_arr, fy_arr, cx_arr, cy_arr = self._estimate_intrinsics(
            raw_result, points_np, camera_poses_np, H_target, W_target
        )

        # 5. Build per-frame types
        pf_points = PerFramePointMap(points_np, confidence_np, frame_indices)
        pf_depth = PerFrameDepth(depth_np, confidence_np, frame_indices)
        pf_intrinsics = PerFrameIntrinsics(fx_arr, fy_arr, cx_arr, cy_arr, frame_indices)
        pf_extrinsics = PerFrameExtrinsics(camera_poses_np, frame_indices)

        # 6. Extract RGB from input frames (for BEV coloring)
        rgb = np.stack(
            [np.array(f.convert("RGB")) for f in plain_frames], axis=0,
        )  # (N, H, W, 3) uint8

        recon = Reconstruction(
            points=pf_points,
            depth=pf_depth,
            intrinsics=pf_intrinsics,
            extrinsics=pf_extrinsics,
            metric_scale=metric_scale,
            gravity_direction=gravity_dir,
            rgb=rgb,
            frame_indices=frame_indices,
            segmenter=self._sam3_tool,
            input_images=plain_frames,
            is_video=self._is_video,
        )

        return recon

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Ensure input is a writable numpy array."""
        arr = np.asarray(arr)
        return arr.copy() if not arr.flags.writeable else arr

    @staticmethod
    def _resize_spatial(arr: np.ndarray, H: int, W: int) -> np.ndarray:
        """Resize spatial dimensions of an array to (H, W).

        Supports shapes ``(N, H_in, W_in, C)`` and ``(N, H_in, W_in)``.
        Uses **nearest-neighbor** interpolation to avoid artifacts at depth
        discontinuities and object boundaries.
        """
        from scipy.ndimage import zoom

        H_in, W_in = arr.shape[1], arr.shape[2]
        if (H_in, W_in) == (H, W):
            return arr
        sy, sx = H / H_in, W / W_in
        if arr.ndim == 4:
            # (N, H_in, W_in, C) → (N, H, W, C)
            return zoom(arr, (1, sy, sx, 1), order=0).astype(arr.dtype)
        else:
            # (N, H_in, W_in) → (N, H, W)
            return zoom(arr, (1, sy, sx), order=0).astype(arr.dtype)

    def _find_gravity(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        camera_poses: np.ndarray,
        frames: List[Image.Image],
    ):
        """Find gravity direction via ground plane (SAM3 + RANSAC) or camera down.

        Returns:
            (gravity_dir, rotation, ground_mask) where ground_mask is a
            boolean (H, W) array of detected ground pixels on frame 0,
            or ``None`` if ground detection was not available.
        """
        gravity_dir = np.array([0.0, 1.0, 0.0])  # default: +Y is up
        rotation = None
        ground_mask_out = None  # returned to caller for ground-Y estimation

        # Try SAM3 ground detection on first frame
        try:
            detect_result = self._call_remote_sam3(
                "detect",
                image_source=frames[0],
                prompt="ground floor surface",
            )
            if hasattr(detect_result, "result") and detect_result.result is not None:
                det = detect_result.result
                if hasattr(det, "masks") and det.masks is not None:
                    ground_mask = self._to_numpy(det.masks[0])  # first mask
                    # RANSAC on ground-masked points
                    first_pts = points[0]  # (H, W, 3)
                    first_conf = confidence[0]  # (H, W)
                    # Apply ground mask
                    masked_conf = first_conf.copy()
                    if ground_mask.shape == first_conf.shape:
                        masked_conf[~ground_mask] = 0
                        ground_mask_out = ground_mask

                    plane_normal, _ = GeometryUtils.fit_ground_plane_ransac(
                        first_pts, masked_conf
                    )
                    if plane_normal is not None:
                        # Disambiguate normal sign using camera's "up" direction.
                        # In OpenCV c2w, column 1 is camera +Y = "down" in world.
                        cam_down = camera_poses[0][:3, 1]
                        cam_up = -cam_down
                        if np.dot(plane_normal, cam_up) < 0:
                            plane_normal = -plane_normal
                        target_up = np.array([0.0, 1.0, 0.0])
                        rotation = GeometryUtils.rotation_matrix_from_vectors(
                            plane_normal, target_up
                        )
                        gravity_dir = rotation @ plane_normal
                        return gravity_dir, rotation, ground_mask_out
        except Exception:
            pass

        # Fallback: use first camera's "down" direction
        try:
            c2w = camera_poses[0]  # (4, 4)
            R = c2w[:3, :3]
            cam_down = R @ np.array([0.0, 1.0, 0.0])  # +Y in camera (OpenCV) = down
            # cam_down is the gravity direction → align to -Y (down in aligned frame)
            # so that +Y = up
            target_down = np.array([0.0, -1.0, 0.0])
            rotation = GeometryUtils.rotation_matrix_from_vectors(cam_down, target_down)
            gravity_dir = np.array([0.0, 1.0, 0.0])
        except Exception:
            rotation = None
            gravity_dir = np.array([0.0, 1.0, 0.0])

        return gravity_dir, rotation, ground_mask_out

    @staticmethod
    def _estimate_ground_y(
        points: np.ndarray,
        confidence: np.ndarray,
        ground_mask: Optional[np.ndarray],
        conf_threshold: float = 0.3,
    ) -> Optional[float]:
        """Estimate the Y-coordinate of the ground plane after gravity alignment.

        With +Y=up convention, ground is at the lowest Y values (percentile 5).
        """
        if ground_mask is not None:
            first_pts = points[0]
            first_conf = confidence[0]
            valid = ground_mask & (first_conf > conf_threshold)
            if valid.sum() > 50:
                y_vals = first_pts[valid, 1]
                return float(np.median(y_vals))

        all_y = []
        for i in range(points.shape[0]):
            valid = confidence[i] > conf_threshold
            if valid.sum() > 0:
                all_y.append(points[i][valid, 1])
        if len(all_y) == 0:
            return None
        all_y = np.concatenate(all_y)
        if len(all_y) < 100:
            return None
        return float(np.percentile(all_y, 5))

    @staticmethod
    def _align_forward_to_neg_z(
        points: np.ndarray,
        camera_poses: np.ndarray,
    ) -> None:
        """Rotate the scene around Y so the first camera looks toward -Z.

        Only applies a yaw rotation (around the Y axis) so gravity alignment
        is preserved.  Modifies *points* and *camera_poses* in-place.
        """
        N = points.shape[0]
        # First camera forward direction (column 2 of c2w rotation = camera -Z in OpenCV)
        fwd = camera_poses[0][:3, 2]
        # Project onto XZ plane (ignore Y / gravity component)
        fwd_xz = np.array([fwd[0], fwd[2]], dtype=np.float64)
        norm = np.linalg.norm(fwd_xz)
        if norm < 1e-6:
            return  # camera looking straight up/down, skip
        fwd_xz /= norm

        # Target: -Z direction in XZ plane = [0, -1]
        # Yaw angle from current forward to target -Z
        # fwd_xz = [sin(θ), cos(θ)] where θ is angle from +Z
        # We want fwd_xz → [0, -1], so rotation angle = atan2(fwd_xz[0], fwd_xz[1]) + π
        cos_a = -fwd_xz[1]  # dot([0,-1], fwd_xz) = -fwd_xz[1]
        sin_a = fwd_xz[0]   # cross([0,-1], fwd_xz) = fwd_xz[0]

        # Yaw rotation matrix around Y axis
        yaw = np.eye(3, dtype=np.float64)
        yaw[0, 0] = cos_a
        yaw[0, 2] = sin_a
        yaw[2, 0] = -sin_a
        yaw[2, 2] = cos_a

        yaw4 = np.eye(4, dtype=np.float64)
        yaw4[:3, :3] = yaw

        for i in range(N):
            flat = points[i].reshape(-1, 3)
            points[i] = (yaw @ flat.T).T.reshape(points[i].shape)
        for i in range(N):
            camera_poses[i] = yaw4 @ camera_poses[i]

    def _call_remote_sam3(self, method_name: str, **kwargs):
        """Call SAM3 via the SAM3Tool's HTTP-based _call_remote."""
        if self._sam3_tool is None:
            raise RuntimeError("SAM3 handle not available for gravity detection.")
        return self._sam3_tool._call_remote(method_name, **kwargs)

    def _estimate_intrinsics(self, raw_result, points, camera_poses, target_H, target_W):
        """Estimate per-frame intrinsics at the target (output) resolution."""
        N = points.shape[0]
        H, W = target_H, target_W

        # If DA3 provides real intrinsics, use them directly (scaled to target res)
        if hasattr(raw_result, "_intrinsics") and raw_result._intrinsics is not None:
            raw_K = self._to_numpy(raw_result._intrinsics)  # (N, 3, 3)
            H_model = self._to_numpy(raw_result.points).shape[1] if hasattr(raw_result, "points") else H
            W_model = self._to_numpy(raw_result.points).shape[2] if hasattr(raw_result, "points") else W
            # Scale intrinsics from model resolution to target resolution
            sy = H / H_model if H_model != H else 1.0
            sx = W / W_model if W_model != W else 1.0
            fx_arr = raw_K[:, 0, 0].astype(np.float64) * sx
            fy_arr = raw_K[:, 1, 1].astype(np.float64) * sy
            cx_arr = raw_K[:, 0, 2].astype(np.float64) * sx
            cy_arr = raw_K[:, 1, 2].astype(np.float64) * sy
            return fx_arr, fy_arr, cx_arr, cy_arr

        # Default: estimate from image size (common approximation)
        fx_arr = np.full(N, W * 0.8, dtype=np.float64)
        fy_arr = np.full(N, H * 0.8, dtype=np.float64)
        cx_arr = np.full(N, W / 2.0, dtype=np.float64)
        cy_arr = np.full(N, H / 2.0, dtype=np.float64)

        # If raw result has ray directions, compute intrinsics from them
        if hasattr(raw_result, "_rays") and raw_result._rays is not None:
            rays = self._to_numpy(raw_result._rays)  # (N, H_model, W_model, 3)
            H_ray, W_ray = rays.shape[1], rays.shape[2]
            for i in range(N):
                # Rays in camera frame
                c2w = camera_poses[i]
                w2c_rot = np.linalg.inv(c2w[:3, :3])
                r = rays[i].reshape(-1, 3)
                cam_r = (w2c_rot @ r.T).T.reshape(H_ray, W_ray, 3)
                center_ray = cam_r[H_ray // 2, W_ray // 2]
                if abs(center_ray[2]) > 1e-6:
                    edge_ray_x = cam_r[H_ray // 2, -1]
                    edge_ray_y = cam_r[-1, W_ray // 2]
                    fov_x = np.arctan2(abs(edge_ray_x[0]), abs(edge_ray_x[2]))
                    fov_y = np.arctan2(abs(edge_ray_y[1]), abs(edge_ray_y[2]))
                    if fov_x > 0:
                        fx_arr[i] = (W / 2.0) / np.tan(fov_x)
                    if fov_y > 0:
                        fy_arr[i] = (H / 2.0) / np.tan(fov_y)

        return fx_arr, fy_arr, cx_arr, cy_arr

    def __repr__(self) -> str:
        return f"ReconstructTool(MAX_FRAMES={self.MAX_FRAMES})"
