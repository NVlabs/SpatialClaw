"""Per-frame typed data classes.

ALL array fields are ``numpy.ndarray`` (never ``torch.Tensor``).
Tools convert tensors to numpy internally before returning.

Every class carries ``frame_indices`` and enforces alignment via
``validate_alignment()``.
"""

from typing import List, Optional

import numpy as np
from PIL import Image

from spatial_agent.kernel_types.visual_feedback import VisualFeedback


class PerFrameData:
    """Base class for all per-frame data."""

    def __init__(self, frame_indices: List[int]):
        self._frame_indices = list(frame_indices)

    @property
    def frame_indices(self) -> List[int]:
        return self._frame_indices

    @property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    def validate_alignment(self, other: "PerFrameData", strict: bool = True) -> None:
        """Check that frame indices match between two PerFrameData objects.

        Raises ``ValueError`` with a clear message on mismatch.
        """
        if strict and self._frame_indices != other._frame_indices:
            raise ValueError(
                f"Frame index mismatch: {self._frame_indices} vs "
                f"{other._frame_indices}. Inputs must have identical "
                f"frame_indices for alignment."
            )

    def __iter__(self):
        raise TypeError(
            f"Cannot iterate over {self.__class__.__name__} directly. "
            f"Use: for fi in obj.frame_indices: data = obj[fi]"
        )

    def __len__(self):
        return self.num_frames

    def get_by_frame_index(self, abs_frame_idx: int) -> int:
        """Return the local index for an absolute frame index."""
        if not isinstance(abs_frame_idx, (int, np.integer)):
            raise TypeError(
                f"Frame index must be an integer, got {type(abs_frame_idx).__name__}: {abs_frame_idx!r}.\n"
                f"  Single frame: obj[frame_idx]\n"
                f"  Batch array: obj.camera_poses (extrinsics), obj.points (point map), obj.depth (depth)"
            )
        if abs_frame_idx not in self._frame_indices:
            avail = self._frame_indices
            hint = ""
            if abs_frame_idx == 0 and 0 not in avail:
                hint = (
                    f"\n  Hint: Did you mean frame={avail[0]}? "
                    f"Use ABSOLUTE frame indices (from seg.frame_indices), not 0-based local indices."
                )
            raise KeyError(
                f"Frame {abs_frame_idx} not found in {avail}. "
                f"Available frames: {avail}{hint}"
            )
        return self._frame_indices.index(abs_frame_idx)

    def __repr__(self) -> str:
        fi = self._frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        return f"{self.__class__.__name__}(num_frames={self.num_frames}, frames={fi_str})"


class PerFrameDepth(PerFrameData):
    """Per-frame depth maps.

    Attributes:
        depth: ``np.ndarray`` of shape ``(N, H, W)`` float32.
        confidence: ``np.ndarray`` of shape ``(N, H, W)`` float32 in [0, 1].
    """

    def __init__(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        frame_indices: List[int],
    ):
        super().__init__(frame_indices)
        assert depth.shape[0] == len(frame_indices), (
            f"depth has {depth.shape[0]} frames but frame_indices has {len(frame_indices)}"
        )
        self.depth = np.asarray(depth, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Get ``(H, W)`` depth map by absolute frame index."""
        local = self.get_by_frame_index(frame_idx)
        return self.depth[local]

    def __repr__(self) -> str:
        fi = self._frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        return (
            f"PerFrameDepth(frames={fi_str}, shape={self.depth.shape})\n"
            f"  Use: recon.depth[frame_idx]  # returns (H, W) float32"
        )


class PerFrameMask(PerFrameData):
    """Per-frame segmentation masks.

    Attributes:
        masks: ``np.ndarray`` of shape ``(N, N_obj, H, W)`` bool.
        labels: Label per object.
        object_ids: Unique ID per object.
    """

    def __init__(
        self,
        masks: np.ndarray,
        labels: List[str],
        object_ids: List[int],
        frame_indices: List[int],
        frames: Optional[dict] = None,
    ):
        super().__init__(frame_indices)
        assert masks.shape[0] == len(frame_indices)
        self.masks = np.asarray(masks, dtype=bool)
        self.labels = labels
        self.object_ids = object_ids
        self._frames = frames or {}  # frame_index → PIL.Image

    @property
    def num_objects(self) -> int:
        return self.masks.shape[1]

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Get ``(N_obj, H, W)`` bool masks by absolute frame index."""
        local = self.get_by_frame_index(frame_idx)
        return self.masks[local]

    def get_mask(self, frame: int, object: "int | str" = 0) -> np.ndarray:
        """Get ``(H, W)`` bool mask by absolute frame index and object index/label.

        Args:
            frame: Absolute frame index.
            object: Object index (int) or label string to match against ``self.labels``.

        Returns:
            ``(H, W)`` boolean mask.
        """
        local = self.get_by_frame_index(frame)
        obj_idx = self._resolve_object(object)
        return self.masks[local, obj_idx]

    def get_masked_points(
        self, recon: "Reconstruction", frame: int, object: "int | str" = 0
    ) -> np.ndarray:
        """Get ``(K, 3)`` world-coordinate points under mask.

        Args:
            recon: ``Reconstruction`` object (provides point maps).
            frame: Absolute frame index.
            object: Object index (int) or label string.

        Returns:
            ``(K, 3)`` float32 array of 3D points under the mask.
        """
        mask = self.get_mask(frame, object)
        local_r = recon.points.get_by_frame_index(frame)
        return recon.points.points[local_r][mask]

    def get_centroid_3d(
        self, recon: "Reconstruction", frame: int, object: "int | str" = 0,
        conf_threshold: float = 0.3,
    ) -> np.ndarray:
        """Get ``(3,)`` median 3D position of confidence-filtered masked points.

        Args:
            recon: ``Reconstruction`` object (provides point maps).
            frame: Absolute frame index.
            object: Object index (int) or label string.
            conf_threshold: Minimum point-map confidence to include (default 0.3).

        Returns:
            ``(3,)`` float32 array. Returns ``[nan, nan, nan]`` if the mask
            is empty (safe for arithmetic; use ``np.isnan()`` to check).
        """
        mask_2d = self.get_mask(frame, object)
        local_r = recon.points.get_by_frame_index(frame)
        pts = recon.points.points[local_r][mask_2d]
        if len(pts) == 0:
            obj_label = object if isinstance(object, str) else self.labels[self._resolve_object(object)]
            print(f"[WARNING] get_centroid_3d: mask for '{obj_label}' at frame {frame} is empty — returning [nan, nan, nan]")
            return np.array([float('nan')] * 3, dtype=np.float32)
        # Filter by reconstruction confidence
        conf = recon.points.confidence[local_r][mask_2d]
        valid = conf > conf_threshold
        if valid.sum() > 0:
            pts = pts[valid]
        return np.median(pts, axis=0)

    def _resolve_object(self, object: "int | str") -> int:
        """Resolve object index from int or label string."""
        if isinstance(object, (int, np.integer)):
            object = int(object)
            if object < 0 or object >= self.num_objects:
                raise IndexError(
                    f"Object index {object} out of range. "
                    f"Available: 0..{self.num_objects - 1} ({self.labels})"
                )
            return object
        if isinstance(object, str):
            for i, label in enumerate(self.labels):
                if label == object:
                    return i
            raise KeyError(
                f"Object label {object!r} not found. Available: {self.labels}"
            )
        raise TypeError(f"`object` must be int or str, got {type(object).__name__}")

    def visualize(
        self,
        frame_idx=None,
        background: Optional[Image.Image] = None,
        **kwargs,
    ) -> VisualFeedback:
        """Render mask overlay for a single frame, returning VisualFeedback.

        Translucent coloured masks with white contour borders and ID labels.

        Args:
            frame_idx: Absolute frame index. Also accepts ``frame=`` as alias.
            background: Background image to overlay masks on.  Falls back to
                stored frames from segmentation if not provided.
        """
        if frame_idx is None and 'frame' in kwargs:
            frame_idx = kwargs.pop('frame')
        if frame_idx is None:
            raise TypeError("visualize() requires a frame index: seg.visualize(31) or seg.visualize(frame=31)")
        if kwargs:
            raise TypeError(f"visualize() got unexpected keyword arguments: {list(kwargs.keys())}")
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local = self.get_by_frame_index(frame_idx)
        frame_masks = self.masks[local]  # (N_obj, H, W)
        n_obj = frame_masks.shape[0]
        H, W = frame_masks.shape[1], frame_masks.shape[2]

        # Resolve background: explicit arg > stored frame > gray placeholder
        if background is None:
            background = self._frames.get(frame_idx)
        if background is not None:
            bg_img = background.convert("RGB")
            # Resize background to match mask resolution if they differ
            if bg_img.height != H or bg_img.width != W:
                bg_img = bg_img.resize((W, H), Image.LANCZOS)
            bg_arr = np.asarray(bg_img)
        else:
            bg_arr = np.full((H, W, 3), 128, dtype=np.uint8)

        # Set figure size to match image aspect ratio for zero-margin rendering
        dpi = 100
        fig_w, fig_h = W / dpi, H / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_position([0, 0, 1, 1])

        ax.imshow(bg_arr)

        if n_obj == 0:
            ax.axis("off")
            fig.canvas.draw()
            img = Image.frombytes(
                "RGBA",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
            ).convert("RGB")
            plt.close(fig)
            return VisualFeedback(
                image=img,
                source="PerFrameMask.visualize",
                description=f"Segmentation masks for frame {frame_idx}: {self.labels}",
                frame_index=frame_idx,
            )

        cmap = plt.cm.get_cmap("gist_rainbow", max(n_obj, 1))

        for i in range(n_obj):
            mask = frame_masks[i]  # (H, W) bool
            if not mask.any():
                continue  # skip empty masks (failed detections)
            color_rgb = np.array(cmap(i)[:3])
            color_rgba = np.array([*color_rgb, 0.5])

            # Translucent fill
            mask_img = mask.reshape(*mask.shape, 1) * color_rgba.reshape(1, 1, -1)
            ax.imshow(mask_img)

            # White contour border
            mask_u8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            border_img = np.zeros((*mask.shape, 4), dtype=np.float32)
            border_img = cv2.drawContours(border_img, contours, -1, (1, 1, 1, 0.8), thickness=2)
            ax.imshow(border_img)

            # ID label at mask centroid
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = float(ys.mean()), float(xs.mean())
                text = f"{self.object_ids[i]}: {self.labels[i]}"
                ax.text(
                    cx, cy, text,
                    fontsize=11, color="white", ha="center", va="center",
                    bbox=dict(facecolor=color_rgb, alpha=0.75, boxstyle="round,pad=0.2"),
                )

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        fig.canvas.draw()
        img = Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        ).convert("RGB")
        plt.close(fig)

        return VisualFeedback(
            image=img,
            source="PerFrameMask.visualize",
            description=f"Segmentation masks for frame {frame_idx}: {self.labels}",
            frame_index=frame_idx,
        )

    def __repr__(self) -> str:
        fi = self._frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        return (
            f"PerFrameMask(frames={fi_str}, objects={self.labels}, shape={self.masks.shape})\n"
            f"  Use: seg.get_mask(frame={fi[0]}, object=0)  # (H, W) bool\n"
            f"  Use: seg.get_centroid_3d(recon, frame={fi[0]}, object=0)  # (3,) float32\n"
            f"  Use: seg.visualize({fi[0]})  # VisualFeedback"
        )


class PerFramePointMap(PerFrameData):
    """Per-frame 3D point maps.

    Attributes:
        points: ``np.ndarray`` of shape ``(N, H, W, 3)`` float32 (XYZ world).
        confidence: ``np.ndarray`` of shape ``(N, H, W)`` float32.
    """

    def __init__(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        frame_indices: List[int],
    ):
        super().__init__(frame_indices)
        assert points.shape[0] == len(frame_indices)
        self.points = np.asarray(points, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Get ``(H, W, 3)`` point map by absolute frame index."""
        local = self.get_by_frame_index(frame_idx)
        return self.points[local]

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base[:-1]}, shape={self.points.shape})"


class PerFrameIntrinsics(PerFrameData):
    """Per-frame camera intrinsics.

    Attributes:
        fx, fy, cx, cy: ``np.ndarray`` each of shape ``(N,)`` float64.
    """

    def __init__(
        self,
        fx: np.ndarray,
        fy: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        frame_indices: List[int],
    ):
        super().__init__(frame_indices)
        self.fx = np.asarray(fx, dtype=np.float64)
        self.fy = np.asarray(fy, dtype=np.float64)
        self.cx = np.asarray(cx, dtype=np.float64)
        self.cy = np.asarray(cy, dtype=np.float64)

    def __getitem__(self, frame_idx: int) -> dict:
        """Get intrinsics dict by absolute frame index.

        Returns:
            ``dict(fx=float, fy=float, cx=float, cy=float)``
        """
        local = self.get_by_frame_index(frame_idx)
        return dict(
            fx=float(self.fx[local]),
            fy=float(self.fy[local]),
            cx=float(self.cx[local]),
            cy=float(self.cy[local]),
        )


class PerFrameExtrinsics(PerFrameData):
    """Per-frame camera extrinsics (camera-to-world SE(3)).

    Attributes:
        camera_poses: ``np.ndarray`` of shape ``(N, 4, 4)`` float64.
    """

    def __init__(
        self,
        camera_poses: np.ndarray,
        frame_indices: List[int],
    ):
        super().__init__(frame_indices)
        assert camera_poses.shape[0] == len(frame_indices)
        self.camera_poses = np.asarray(camera_poses, dtype=np.float64)

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Get ``(4, 4)`` camera-to-world pose by absolute frame index."""
        local = self.get_by_frame_index(frame_idx)
        return self.camera_poses[local]


class PerFrameCoordinates(PerFrameData):
    """Per-frame 2D/3D coordinates for tracked points.

    Attributes:
        coords_2d: ``np.ndarray`` of shape ``(N, K, 2)`` float64, pixel coords.
        coords_3d: ``np.ndarray`` of shape ``(N, K, 3)`` float64, world coords.
        labels: K labels for each tracked point.
    """

    def __init__(
        self,
        coords_2d: np.ndarray,
        coords_3d: np.ndarray,
        labels: List[str],
        frame_indices: List[int],
    ):
        super().__init__(frame_indices)
        self.coords_2d = np.asarray(coords_2d, dtype=np.float64)
        self.coords_3d = np.asarray(coords_3d, dtype=np.float64)
        self.labels = labels


class Reconstruction(PerFrameData):
    """Complete reconstruction output, gravity-aligned.

    All arrays are numpy.  Contains per-frame depth, intrinsics, extrinsics,
    and point maps, plus a BEV visualization as ``VisualFeedback``.

    BEV Rendering (``render_bev``):
        Produces a top-down BEV with object annotations (oriented
        bounding boxes for stationary objects, red→blue trajectories
        for moving objects) and optional camera egomotion trajectory
        (green→yellow dashed path via ``ego_trajectory=True``).

        Accepts text *prompts* for on-the-fly SAM3 segmentation, or
        pre-computed *masks*.  The reference camera's forward direction
        is aligned to the plot's upward direction (configurable via
        ``ref_frame``).
    """

    def __init__(
        self,
        points: PerFramePointMap,
        depth: PerFrameDepth,
        intrinsics: PerFrameIntrinsics,
        extrinsics: PerFrameExtrinsics,
        metric_scale: float,
        gravity_direction: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        frame_indices: Optional[List[int]] = None,
        segmenter=None,
        input_images=None,
        is_video: bool = True,
    ):
        fi = frame_indices or points.frame_indices
        super().__init__(fi)
        self.points = points
        self.depth = depth
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.metric_scale = metric_scale
        # Stores the "up" direction [0,1,0].  +Y = up in the aligned frame.
        # First camera forward is aligned to -Z.
        self.gravity_direction = np.asarray(gravity_direction, dtype=np.float64)
        self._rgb = rgb  # (N, H, W, 3) uint8 — from input frames
        self._segmenter = segmenter    # duck-typed SAM3Tool (no import needed)
        self._is_video = is_video
        self._input_images = input_images  # PIL images for single-view / prompts

    # ------------------------------------------------------------------
    # BEV rendering
    # ------------------------------------------------------------------

    def render_bev(
        self,
        masks: Optional["PerFrameMask"] = None,
        labels: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.1,
        ref_frame: int = 0,
        ego_trajectory: bool = True,
    ) -> VisualFeedback:
        """Render a bird's-eye view of the reconstruction.

        Produces a top-down BEV showing object annotations and the camera
        egomotion trajectory by default.  No pointcloud scatter is drawn.

        Objects are annotated with smart motion detection: stationary
        objects get oriented bounding boxes, moving objects get
        colour-graded trajectories (red → blue with arrowhead).

        Args:
            masks: Pre-computed per-frame masks (``PerFrameMask`` or raw
                ``(N, N_obj, H, W)`` bool array).
            labels: Object labels (required when *masks* is a raw array).
            prompts: Text prompts for on-the-fly segmentation via SAM3.
                Ignored when *masks* is already provided.
            conf_threshold: Minimum point-map confidence for object BEV
                projection.
            ref_frame: Absolute frame index (from ``recon.frame_indices``)
                whose camera forward direction defines the plot's upward
                direction.  Default 0 uses the first reconstructed frame.
            ego_trajectory: If True, draw the camera's egomotion path as a
                green→yellow dashed trajectory.

        Returns:
            ``VisualFeedback`` containing the rendered BEV image.
        """
        # Auto-segment from prompts if no masks provided
        if prompts is not None and masks is None:
            masks = self._segment_from_prompts(prompts)
            if hasattr(masks, "labels"):
                labels = masks.labels

        return self._render_multi_view(masks, labels, conf_threshold, ref_frame, ego_trajectory)

    def _segment_from_prompts(self, prompts: List[str]) -> "PerFrameMask":
        """Segment objects using text prompts via the attached segmenter.

        Single-view uses ``segment_image``; multi-view with a video source
        uses ``segment_video``; multi-view from static images segments each
        frame independently and stacks the results.
        """
        if self._segmenter is None:
            raise RuntimeError(
                "No segmenter attached to this Reconstruction. "
                "Pass prompts= only when Pi3 was constructed with a SAM3 "
                "tool, or provide pre-computed masks= instead."
            )
        if self.num_frames == 1:
            if self._input_images is None or len(self._input_images) == 0:
                raise RuntimeError(
                    "Cannot segment: no input images stored in "
                    "Reconstruction."
                )
            return self._segmenter.segment_image(
                self._input_images[0],
                prompts,
                frame_index=self.frame_indices[0],
            )
        else:
            # Try video segmentation first (requires video_source)
            has_video = getattr(self._segmenter, "_video_source", None) is not None
            if has_video:
                return self._segmenter.segment_video(
                    prompts, frame_indices=self.frame_indices,
                )
            # Fallback: segment each static image independently and stack
            if self._input_images is None or len(self._input_images) == 0:
                raise RuntimeError(
                    "Cannot segment: no input images or video source "
                    "available in Reconstruction."
                )
            per_frame_masks = []
            labels = None
            for img, fidx in zip(self._input_images, self.frame_indices):
                pfm = self._segmenter.segment_image(
                    img, prompts, frame_index=fidx,
                )
                per_frame_masks.append(pfm.masks)  # (1, N_obj, H, W)
                if labels is None:
                    labels = pfm.labels
            # Stack along frame dim: (N_frames, N_obj, H, W)
            stacked = np.concatenate(per_frame_masks, axis=0)
            return PerFrameMask(
                masks=stacked,
                labels=labels,
                object_ids=list(range(len(prompts))),
                frame_indices=self.frame_indices,
            )

    def _render_single_view(
        self,
        masks,
        labels: Optional[List[str]],
    ) -> VisualFeedback:
        """Render 2D segmentation overlay for a single-frame reconstruction.

        Delegates to ``PerFrameMask.visualize()`` with the input image as
        background.
        """
        bg = None
        if self._input_images is not None and len(self._input_images) > 0:
            bg = self._input_images[0]

        if hasattr(masks, "visualize"):
            return masks.visualize(masks.frame_indices[0], background=bg)

        # Raw ndarray masks → wrap in PerFrameMask
        mask_arr = np.asarray(masks, dtype=bool)
        if labels is None:
            labels = [f"obj_{i}" for i in range(mask_arr.shape[1])]
        pfm = PerFrameMask(
            masks=mask_arr,
            labels=labels,
            object_ids=list(range(len(labels))),
            frame_indices=self.frame_indices[:1],
        )
        return pfm.visualize(self.frame_indices[0], background=bg)

    def _render_multi_view(
        self,
        masks,
        labels: Optional[List[str]],
        conf_threshold: float = 0.1,
        ref_frame: int = 0,
        ego_trajectory: bool = True,
    ) -> VisualFeedback:
        """Render a bird's-eye view of a multi-frame reconstruction.

        The reference camera's forward direction is aligned to the plot's
        upward direction.  Objects are annotated with smart motion detection:
        stationary → oriented bbox, moving → colour-graded trajectory.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points = self.points.points            # (N, H, W, 3)
        confidence = self.points.confidence    # (N, H, W)
        camera_poses = self.extrinsics.camera_poses  # (N, 4, 4)

        # -- Reference camera alignment --
        # ref_frame is an absolute frame index; convert to local array index.
        # Fallback: if the absolute index is not found, use local 0.
        if ref_frame in self._frame_indices:
            ref = self._frame_indices.index(ref_frame)
        else:
            ref = max(0, min(ref_frame, len(camera_poses) - 1))
        cam_fwd_3d = camera_poses[ref][:3, 2]
        fwd_xz = np.array([cam_fwd_3d[0], cam_fwd_3d[2]], dtype=np.float64)
        fwd_norm = np.linalg.norm(fwd_xz)
        if fwd_norm > 1e-6:
            fwd_xz /= fwd_norm
        else:
            fwd_xz = np.array([0.0, 1.0])
        # Camera image-right in world XZ: CCW 90° rotation of forward so that
        # the plot's +u axis aligns with pose[:3,0] (camera +X = image-right).
        right_xz = np.array([-fwd_xz[1], fwd_xz[0]])

        # Sanity: right_xz must agree with the camera's image-right axis
        # (pose[:3, 0]) projected onto XZ. Catches future sign-of-rotation regressions.
        cam_right_xz = np.array([camera_poses[ref][0, 0], camera_poses[ref][2, 0]], dtype=np.float64)
        cam_right_norm = np.linalg.norm(cam_right_xz)
        if cam_right_norm > 1e-3:
            if np.dot(right_xz, cam_right_xz / cam_right_norm) < 0.0:
                raise AssertionError(
                    "BEV right_xz sign disagrees with pose[:3, 0] image-right axis. "
                    "This indicates a regression in the BEV axis convention."
                )

        # -- Draw --
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_facecolor("#f0f0f0")

        # -- Smart object annotations --
        if masks is not None:
            self._draw_bev_objects(
                ax, masks, labels, points, confidence,
                right_xz, fwd_xz, conf_threshold,
            )

        # -- Initial span (cameras + objects) for sizing camera wedges --
        positions = camera_poses[:, :3, 3]
        cam_u_arr = np.array([float(np.dot([p[0], p[2]], right_xz)) for p in positions])
        cam_v_arr = np.array([float(np.dot([p[0], p[2]], fwd_xz)) for p in positions])
        init_u = list(cam_u_arr)
        init_v = list(cam_v_arr)
        dl0 = ax.dataLim
        if np.isfinite(dl0.x0) and np.isfinite(dl0.x1):
            init_u.extend([dl0.x0, dl0.x1])
            init_v.extend([dl0.y0, dl0.y1])
        span_init = max(
            max(init_u) - min(init_u),
            max(init_v) - min(init_v),
            1.0,
        )

        # -- Egomotion trajectory (video) or per-camera wedges (multi-image) --
        if ego_trajectory:
            if self._is_video:
                Reconstruction._draw_ego_trajectory(
                    ax, camera_poses, right_xz, fwd_xz,
                )
            else:
                Reconstruction._draw_camera_markers(
                    ax, camera_poses, right_xz, fwd_xz,
                    self._frame_indices, ref, span_init,
                )

        # -- Reference camera wedge (FOV-shaped, oriented to its forward dir) --
        cam_pos = camera_poses[ref][:3, 3]
        cam_u = float(np.dot([cam_pos[0], cam_pos[2]], right_xz))
        cam_v = float(np.dot([cam_pos[0], cam_pos[2]], fwd_xz))
        ref_fwd_3d = camera_poses[ref][:3, 2]
        ref_f_u = float(np.dot([ref_fwd_3d[0], ref_fwd_3d[2]], right_xz))
        ref_f_v = float(np.dot([ref_fwd_3d[0], ref_fwd_3d[2]], fwd_xz))
        ref_f_norm = float(np.hypot(ref_f_u, ref_f_v))
        if ref_f_norm > 1e-9:
            ref_f_u /= ref_f_norm
            ref_f_v /= ref_f_norm
        else:
            ref_f_u, ref_f_v = 0.0, 1.0
        ref_label = f"Camera {ref}"
        Reconstruction._draw_camera_wedge(
            ax, cam_u, cam_v, ref_f_u, ref_f_v, span_init,
            color="#cc2222", label=ref_label, is_ref=True,
        )
        # Legend proxy for the reference camera (NaN coords keep dataLim clean)
        from matplotlib.lines import Line2D as _L2D
        ax.add_line(_L2D(
            [np.nan], [np.nan], marker="^", color="#cc2222",
            markersize=12, linestyle="None",
            markeredgecolor="white", markeredgewidth=1.2,
            label=ref_label,
        ))

        # -- Final bounds (camera wedges now contribute via dataLim) --
        bound_u = list(cam_u_arr)
        bound_v = list(cam_v_arr)
        dl = ax.dataLim
        if np.isfinite(dl.x0) and np.isfinite(dl.x1):
            bound_u.extend([dl.x0, dl.x1])
            bound_v.extend([dl.y0, dl.y1])

        u_min, u_max = min(bound_u), max(bound_u)
        v_min, v_max = min(bound_v), max(bound_v)
        span = max(u_max - u_min, v_max - v_min, 1.0)
        pad = span * 0.15
        cx_b = (u_min + u_max) / 2
        cy_b = (v_min + v_max) / 2
        half = span / 2 + pad
        ax.set_xlim(cx_b - half, cx_b + half)
        ax.set_ylim(cy_b - half, cy_b + half)

        ax.set_xlabel(r"$\leftarrow$ Left $\cdot\cdot\cdot$ Right $\rightarrow$")
        ax.set_ylabel(f"Camera {ref} Forward " + r"$\rightarrow$")
        ax.set_title("Bird's Eye View (top-down)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # -- Compass-style direction labels at plot edges --
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        compass_style = dict(
            fontsize=14, fontweight="bold", color="#333333",
            zorder=20,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999",
                      boxstyle="round,pad=0.3"),
        )
        ax.text(cx, ylim[1] - (ylim[1] - ylim[0]) * 0.02,
                "FORWARD (away from camera)", **compass_style, ha="center", va="top")
        ax.text(cx, ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                "BEHIND (toward camera)", **compass_style, ha="center", va="bottom")
        ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.02, cy,
                "LEFT", **compass_style, ha="left", va="center")
        ax.text(xlim[1] - (xlim[1] - xlim[0]) * 0.02, cy,
                "RIGHT", **compass_style, ha="right", va="center")

        fig.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), buf,
        ).convert("RGB")
        plt.close(fig)

        desc_parts = [
            "Bird's eye view (top-down)",
            f"camera {ref} is at the bottom looking upward",
            "LEFT/RIGHT/FORWARD/BEHIND labels show camera-relative directions",
        ]
        if masks is not None:
            desc_parts.append(
                "with object annotations (stationary: oriented bounding boxes, "
                "moving: red→blue trajectories)"
            )
        if ego_trajectory:
            if self._is_video:
                desc_parts.append(
                    "camera egomotion path shown as green→yellow dashed trajectory"
                )
            else:
                desc_parts.append(
                    "individual camera positions shown as numbered triangles "
                    "(no connecting trajectory — multi-image input)"
                )

        return VisualFeedback(
            image=img,
            source="Reconstruction.render_bev",
            description=". ".join(desc_parts) + ".",
        )

    # ------------------------------------------------------------------
    # Internal: motion-aware BEV object annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_bev_objects(
        ax,
        masks,
        labels: Optional[List[str]],
        points: np.ndarray,
        confidence: np.ndarray,
        right_xz: np.ndarray,
        fwd_xz: np.ndarray,
        conf_threshold: float,
        erode_px: int = 3,
        mad_k: float = 3.0,
        motion_threshold: float = 0.15,
    ) -> None:
        """Draw motion-aware object annotations on BEV.

        Stationary objects get bounding boxes; moving objects get
        colour-graded centroid trajectories (red → blue with arrowhead).

        Motion is classified by comparing max centroid displacement across
        frames against the object's bounding box diagonal.  A ratio above
        *motion_threshold* (default 0.15) indicates movement.

        Args:
            erode_px: Erosion radius in pixels (0 to disable).
            mad_k: MAD multiplier for outlier rejection.
            motion_threshold: Centroid displacement / bbox diagonal ratio
                above which an object is classified as moving.
        """
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from scipy.ndimage import binary_erosion

        # Resolve mask array and labels
        if hasattr(masks, "masks"):
            mask_arr = masks.masks
            if labels is None:
                labels = masks.labels
        else:
            mask_arr = np.asarray(masks, dtype=bool)

        N_mask, N_obj = mask_arr.shape[0], mask_arr.shape[1]
        H_mask, W_mask = mask_arr.shape[2], mask_arr.shape[3]
        N_pt, H_pt, W_pt = points.shape[0], points.shape[1], points.shape[2]
        N_frames = min(N_mask, N_pt)

        # Erosion structuring element
        if erode_px > 0:
            from scipy.ndimage import generate_binary_structure, iterate_structure
            struct = iterate_structure(
                generate_binary_structure(2, 1), erode_px,
            )
        else:
            struct = None

        cmap = plt.colormaps.get_cmap("tab10")

        for obj_idx in range(N_obj):
            # Collect per-frame point sets
            per_frame_pts: List[np.ndarray] = []
            for t in range(N_frames):
                obj_mask = mask_arr[t, obj_idx]

                if (H_mask, W_mask) != (H_pt, W_pt):
                    from scipy.ndimage import zoom
                    obj_mask = zoom(
                        obj_mask.astype(np.float32),
                        (H_pt / H_mask, W_pt / W_mask),
                        order=0,
                    ) > 0.5

                if struct is not None:
                    eroded = binary_erosion(obj_mask, structure=struct)
                    if eroded.sum() >= obj_mask.sum() * 0.1:
                        obj_mask = eroded

                valid = (confidence[t] > conf_threshold) & obj_mask
                per_frame_pts.append(points[t][valid])

            # Aggregate all frames for bounding box
            obj_pts = (
                np.concatenate(per_frame_pts, axis=0)
                if per_frame_pts
                else np.empty((0, 3))
            )
            if len(obj_pts) < 4:
                continue

            # Project to BEV
            pts_xz = np.column_stack([obj_pts[:, 0], obj_pts[:, 2]])
            u_obj = pts_xz @ right_xz
            v_obj = pts_xz @ fwd_xz

            # MAD-based outlier rejection
            if mad_k > 0 and len(u_obj) > 10:
                keep = np.ones(len(u_obj), dtype=bool)
                for vals in (u_obj, v_obj):
                    med = np.median(vals)
                    mad = np.median(np.abs(vals - med))
                    if mad > 1e-8:
                        keep &= np.abs(vals - med) <= mad_k * mad
                u_obj = u_obj[keep]
                v_obj = v_obj[keep]

            if len(u_obj) < 4:
                continue

            u_lo, u_hi = u_obj.min(), u_obj.max()
            v_lo, v_hi = v_obj.min(), v_obj.max()

            color = cmap(obj_idx % 10)
            label = (
                labels[obj_idx]
                if labels and obj_idx < len(labels)
                else f"obj_{obj_idx}"
            )

            # Classify motion
            is_moving, ratio = Reconstruction._classify_motion(
                per_frame_pts, right_xz, fwd_xz, mad_k,
                u_lo, u_hi, v_lo, v_hi, motion_threshold,
            )

            if is_moving and N_frames > 1:
                # MOVING → trajectory only
                Reconstruction._draw_centroid_trajectory(
                    ax, per_frame_pts, right_xz, fwd_xz,
                    mad_k, N_frames, label,
                )
            else:
                # STATIONARY (or single frame) → minimum-area oriented bbox
                corners = Reconstruction._min_area_bbox(
                    np.column_stack([u_obj, v_obj])
                )
                poly = patches.Polygon(
                    corners, closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    zorder=5,
                )
                ax.add_patch(poly)
                # Place label at the top-most corner
                top_idx = int(np.argmax(corners[:, 1]))
                ax.text(
                    corners[top_idx, 0], corners[top_idx, 1],
                    f" {label}",
                    fontsize=9, color="white", fontweight="bold",
                    verticalalignment="bottom", zorder=6,
                    bbox=dict(
                        facecolor=color, alpha=0.7,
                        boxstyle="round,pad=0.15",
                    ),
                )

    @staticmethod
    def _classify_motion(
        per_frame_pts: List[np.ndarray],
        right_xz: np.ndarray,
        fwd_xz: np.ndarray,
        mad_k: float,
        u_lo: float,
        u_hi: float,
        v_lo: float,
        v_hi: float,
        threshold: float = 0.15,
    ) -> tuple:
        """Classify an object as moving or stationary.

        Computes per-frame median BEV centroids (with MAD filtering),
        then compares max pairwise centroid displacement against the
        bounding box diagonal.

        Returns:
            ``(is_moving, motion_ratio)`` where *motion_ratio* is
            ``max_displacement / bbox_diagonal``.
        """
        bbox_diag = np.sqrt((u_hi - u_lo) ** 2 + (v_hi - v_lo) ** 2)
        if bbox_diag < 1e-8:
            return False, 0.0

        # Compute per-frame centroids
        centroids = []
        for t_pts in per_frame_pts:
            if len(t_pts) < 10:
                continue
            pts_xz = np.column_stack([t_pts[:, 0], t_pts[:, 2]])
            u_t = pts_xz @ right_xz
            v_t = pts_xz @ fwd_xz
            if mad_k > 0 and len(u_t) > 10:
                keep = np.ones(len(u_t), dtype=bool)
                for vals in (u_t, v_t):
                    med = np.median(vals)
                    mad = np.median(np.abs(vals - med))
                    if mad > 1e-8:
                        keep &= np.abs(vals - med) <= mad_k * mad
                u_t, v_t = u_t[keep], v_t[keep]
            if len(u_t) < 4:
                continue
            centroids.append((float(np.median(u_t)), float(np.median(v_t))))

        if len(centroids) < 2:
            return False, 0.0

        # Max pairwise displacement
        pts = np.array(centroids)  # (K, 2)
        max_disp = 0.0
        for i in range(len(pts)):
            diffs = pts[i] - pts[i + 1:]
            if len(diffs) > 0:
                dists = np.sqrt((diffs ** 2).sum(axis=1))
                max_disp = max(max_disp, float(dists.max()))

        ratio = max_disp / bbox_diag
        return ratio > threshold, ratio

    @staticmethod
    def _min_area_bbox(pts_2d: np.ndarray) -> np.ndarray:
        """Compute the minimum-area oriented bounding box for 2D points.

        Uses the rotating calipers approach on the convex hull.

        Args:
            pts_2d: (N, 2) array of 2D points.

        Returns:
            (4, 2) array of corner vertices (ordered).
        """
        from scipy.spatial import ConvexHull

        if len(pts_2d) < 3:
            # Degenerate — fall back to axis-aligned
            lo = pts_2d.min(axis=0)
            hi = pts_2d.max(axis=0)
            return np.array([
                [lo[0], lo[1]], [hi[0], lo[1]],
                [hi[0], hi[1]], [lo[0], hi[1]],
            ])

        try:
            hull = ConvexHull(pts_2d)
        except Exception:
            lo = pts_2d.min(axis=0)
            hi = pts_2d.max(axis=0)
            return np.array([
                [lo[0], lo[1]], [hi[0], lo[1]],
                [hi[0], hi[1]], [lo[0], hi[1]],
            ])

        hull_pts = pts_2d[hull.vertices]  # (K, 2)

        # Try each edge of the convex hull as a candidate base direction
        best_area = np.inf
        best_corners = None

        for i in range(len(hull_pts)):
            edge = hull_pts[(i + 1) % len(hull_pts)] - hull_pts[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            # Unit vectors along and perpendicular to this edge
            u = edge / edge_len
            v = np.array([-u[1], u[0]])

            # Project all hull points onto (u, v)
            proj = hull_pts @ np.column_stack([u, v])  # (K, 2)
            min_uv = proj.min(axis=0)
            max_uv = proj.max(axis=0)

            area = (max_uv[0] - min_uv[0]) * (max_uv[1] - min_uv[1])
            if area < best_area:
                best_area = area
                # Reconstruct 4 corners in original coordinates
                c0 = min_uv[0] * u + min_uv[1] * v
                c1 = max_uv[0] * u + min_uv[1] * v
                c2 = max_uv[0] * u + max_uv[1] * v
                c3 = min_uv[0] * u + max_uv[1] * v
                best_corners = np.array([c0, c1, c2, c3])

        if best_corners is None:
            lo = pts_2d.min(axis=0)
            hi = pts_2d.max(axis=0)
            return np.array([
                [lo[0], lo[1]], [hi[0], lo[1]],
                [hi[0], hi[1]], [lo[0], hi[1]],
            ])

        return best_corners

    @staticmethod
    def _draw_centroid_trajectory(
        ax,
        per_frame_pts: List[np.ndarray],
        right_xz: np.ndarray,
        fwd_xz: np.ndarray,
        mad_k: float,
        n_total: int,
        label: str,
    ) -> None:
        """Draw a colour-graded centroid trajectory (red → blue) with label.

        For each frame with enough valid points, the **median** BEV
        position is used as the centroid (robust to per-frame depth
        outliers).  Frames with fewer than 10 valid points are skipped.

        The line colour transitions from red (first frame) to blue
        (last frame), with an arrowhead at the final position.
        A label is placed at the trajectory midpoint for identification.
        """
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        centroids = []
        for t, t_pts in enumerate(per_frame_pts):
            if len(t_pts) < 10:
                continue
            pts_xz = np.column_stack([t_pts[:, 0], t_pts[:, 2]])
            u_t = pts_xz @ right_xz
            v_t = pts_xz @ fwd_xz
            if mad_k > 0 and len(u_t) > 10:
                keep = np.ones(len(u_t), dtype=bool)
                for vals in (u_t, v_t):
                    med = np.median(vals)
                    mad = np.median(np.abs(vals - med))
                    if mad > 1e-8:
                        keep &= np.abs(vals - med) <= mad_k * mad
                u_t, v_t = u_t[keep], v_t[keep]
            if len(u_t) < 4:
                continue
            centroids.append((t, float(np.median(u_t)), float(np.median(v_t))))

        if len(centroids) < 2:
            return

        indices = np.array([c[0] for c in centroids], dtype=float)
        pts = np.array([[c[1], c[2]] for c in centroids])

        t_norm = indices / max(n_total - 1, 1)

        segments = np.array(
            [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        )
        seg_t = (t_norm[:-1] + t_norm[1:]) / 2

        # MAD-based temporal outlier filter on inter-frame step lengths.
        # Breaks the line where the centroid jumps far relative to its own
        # typical step (occlusion-and-reappear, mask-on-wrong-object, etc.).
        # Scale-invariant: uses the trajectory's own median/MAD.
        n_seg = len(pts) - 1
        if n_seg >= 5:
            step_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            med_step = float(np.median(step_lengths))
            mad_step = float(np.median(np.abs(step_lengths - med_step)))
            # MAD rule covers normal distributions; ratio rule covers tight
            # distributions where MAD≈0. Outlier if it exceeds either.
            threshold = max(med_step + 5.0 * mad_step, 5.0 * med_step, 1e-9)
            seg_mask = step_lengths <= threshold
        else:
            seg_mask = np.ones(max(n_seg, 0), dtype=bool)

        cmap_rb = LinearSegmentedColormap.from_list("rb", ["red", "blue"])

        if seg_mask.any():
            lc = LineCollection(
                segments[seg_mask], cmap=cmap_rb, linewidths=3.5, zorder=7,
                alpha=0.9,
            )
            lc.set_array(seg_t[seg_mask])
            lc.set_clim(0, 1)
            ax.add_collection(lc)

        # Find the last non-outlier segment so the arrowhead and direction
        # text reflect a trustworthy endpoint.
        last_good_seg = None
        for i in range(n_seg - 1, -1, -1):
            if seg_mask[i]:
                last_good_seg = i
                break
        last_pt_idx = (last_good_seg + 1) if last_good_seg is not None else 0

        # Arrowhead at the last non-outlier segment
        if last_good_seg is not None:
            ax.annotate(
                "",
                xy=(pts[last_good_seg + 1, 0], pts[last_good_seg + 1, 1]),
                xytext=(pts[last_good_seg, 0], pts[last_good_seg, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=cmap_rb(float(t_norm[last_good_seg + 1])),
                    lw=3.5,
                    mutation_scale=30,
                ),
                zorder=8,
            )

        # Start marker (red dot)
        ax.plot(
            pts[0, 0], pts[0, 1], "o",
            color=cmap_rb(float(t_norm[0])),
            markersize=8, zorder=8,
        )

        # Camera-relative direction of motion (use last trustworthy endpoint).
        disp_u = pts[last_pt_idx, 0] - pts[0, 0]  # positive = rightward
        disp_v = pts[last_pt_idx, 1] - pts[0, 1]  # positive = forward
        dir_parts = []
        if abs(disp_v) > abs(disp_u) * 0.3:
            dir_parts.append("forward" if disp_v > 0 else "backward")
        if abs(disp_u) > abs(disp_v) * 0.3:
            dir_parts.append("right" if disp_u > 0 else "left")
        dir_text = "+".join(dir_parts) if dir_parts else "stationary"

        # Label at trajectory midpoint with direction
        mid_idx = len(pts) // 2
        mid_t = t_norm[mid_idx]
        ax.text(
            pts[mid_idx, 0], pts[mid_idx, 1],
            f" {label} → {dir_text}",
            fontsize=9, color="white", fontweight="bold",
            verticalalignment="center", zorder=9,
            bbox=dict(
                facecolor=cmap_rb(float(mid_t)),
                alpha=0.8,
                boxstyle="round,pad=0.2",
            ),
        )

    @staticmethod
    def _draw_ego_trajectory(
        ax,
        camera_poses: np.ndarray,
        right_xz: np.ndarray,
        fwd_xz: np.ndarray,
    ) -> None:
        """Draw camera egomotion trajectory as a green→yellow dashed path.

        Each camera position is projected to BEV and connected with a
        colour-graded line (green = first frame → yellow = last frame).
        """
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        N = len(camera_poses)
        if N < 2:
            return

        positions = camera_poses[:, :3, 3]  # (N, 3)
        xz = np.column_stack([positions[:, 0], positions[:, 2]])
        u = xz @ right_xz
        v = xz @ fwd_xz
        pts = np.column_stack([u, v])

        t_norm = np.linspace(0, 1, N)

        segments = np.array([[pts[i], pts[i + 1]] for i in range(N - 1)])
        seg_t = (t_norm[:-1] + t_norm[1:]) / 2

        cmap_ego = LinearSegmentedColormap.from_list("ego", ["#22cc22", "#cccc22"])

        lc = LineCollection(
            segments, cmap=cmap_ego, linewidths=2.5, zorder=4, alpha=0.8,
            linestyle="--",
        )
        lc.set_array(seg_t)
        lc.set_clim(0, 1)
        ax.add_collection(lc)

        # Start marker (drawn above the reference-camera triangle so the
        # legend's green colour matches what's actually rendered when the
        # reference camera coincides with the ego start)
        ax.plot(u[0], v[0], "o", color="#22cc22", markersize=6, zorder=11,
                markeredgecolor="white", markeredgewidth=0.8,
                label="Ego start")

        # End arrowhead
        ax.annotate(
            "", xy=(u[-1], v[-1]), xytext=(u[-2], v[-2]),
            arrowprops=dict(
                arrowstyle="-|>", color="#cccc22", lw=2.5, mutation_scale=20,
            ),
            zorder=5,
        )

    @staticmethod
    def _draw_camera_wedge(
        ax,
        u: float,
        v: float,
        f_u: float,
        f_v: float,
        span: float,
        color: str,
        label: str = "",
        is_ref: bool = False,
        fov_deg: float = 60.0,
    ) -> None:
        """Draw a camera-from-above wedge marker.

        Apex (camera optical center) is at ``(u, v)``; the FOV wedge opens
        in direction ``(f_u, f_v)``. Reference cameras get a larger, more
        saturated wedge for visual prominence.
        """
        import math
        from matplotlib.patches import Polygon

        scale = 0.10 if is_ref else 0.07
        L = span * scale
        half_w = L * math.tan(math.radians(fov_deg / 2.0))
        # Right-perpendicular to forward (clockwise 90°): (f_v, -f_u)
        r_u, r_v = f_v, -f_u
        apex = (u, v)
        base_l = (u + L * f_u - half_w * r_u, v + L * f_v - half_w * r_v)
        base_r = (u + L * f_u + half_w * r_u, v + L * f_v + half_w * r_v)

        ax.add_patch(Polygon(
            [apex, base_l, base_r], closed=True,
            facecolor=color, edgecolor="white",
            linewidth=1.6 if is_ref else 1.1,
            alpha=0.92 if is_ref else 0.82,
            zorder=11 if is_ref else 9,
        ))
        # Body dot at apex (camera optical center)
        ax.plot(
            u, v, "o", color=color,
            markersize=6 if is_ref else 4,
            markeredgecolor="white",
            markeredgewidth=0.8 if is_ref else 0.6,
            zorder=12 if is_ref else 10,
        )
        if label:
            # Place label slightly behind the apex (opposite to forward)
            off_u, off_v = -f_u, -f_v
            ax.annotate(
                label, xy=(u, v),
                xytext=(int(off_u * 16 - 2), int(off_v * 16 - 2)),
                textcoords="offset points",
                fontsize=13 if is_ref else 11,
                fontweight="bold",
                color=color, zorder=13,
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="none", boxstyle="round,pad=0.25"),
            )

    @staticmethod
    def _draw_camera_markers(
        ax,
        camera_poses: np.ndarray,
        right_xz: np.ndarray,
        fwd_xz: np.ndarray,
        frame_indices: list,
        ref_local: int,
        span: float,
    ) -> None:
        """Draw per-camera FOV wedges for multi-image input (no trajectory).

        Each non-reference camera is shown as a wedge marker oriented in
        its forward direction with a frame-index label. Used when the input
        is a multi-image bag (Metadata.is_video=False), where connecting
        cameras with a trajectory line would be misleading.
        """
        import math
        from matplotlib.lines import Line2D

        N = len(camera_poses)
        if N < 1:
            return

        drew_any = False
        for i in range(N):
            if i == ref_local:
                continue
            pos = camera_poses[i][:3, 3]
            u = float(np.dot([pos[0], pos[2]], right_xz))
            v = float(np.dot([pos[0], pos[2]], fwd_xz))
            fwd_3d = camera_poses[i][:3, 2]
            f_u = float(np.dot([fwd_3d[0], fwd_3d[2]], right_xz))
            f_v = float(np.dot([fwd_3d[0], fwd_3d[2]], fwd_xz))
            f_norm = math.hypot(f_u, f_v)
            if f_norm > 1e-9:
                f_u /= f_norm
                f_v /= f_norm
            else:
                f_u, f_v = 0.0, 1.0
            label = str(frame_indices[i]) if i < len(frame_indices) else str(i)
            Reconstruction._draw_camera_wedge(
                ax, u, v, f_u, f_v, span,
                color="#3366cc", label=label, is_ref=False,
            )
            drew_any = True

        if drew_any:
            # Single legend proxy (NaN-positioned so it doesn't affect dataLim)
            ax.add_line(Line2D(
                [np.nan], [np.nan], marker="^", color="#3366cc",
                markersize=10, linestyle="None",
                markeredgecolor="white", markeredgewidth=0.8,
                label="Cameras",
            ))

    def __repr__(self) -> str:
        fi = self._frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        return (
            f"Reconstruction(frames={fi_str}, num_frames={self.num_frames}, "
            f"metric_scale={self.metric_scale:.4f})\n"
            f"  Use: recon.depth[{fi[0]}]  # (H, W) depth\n"
            f"  Use: recon.extrinsics[{fi[0]}]  # (4, 4) c2w pose\n"
            f"  Use: seg.get_centroid_3d(recon, frame={fi[0]}, object=0)  # (3,) 3D position"
        )
