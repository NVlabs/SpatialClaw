import asyncio
from dataclasses import dataclass
import os
import sys
import tempfile
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, AgentContext, gpu_inference_lock
from spatial_agent.gpu_models.types import SAM3ImageDetectionOutput, SAM3VideoSegmentationOutput  # noqa: F811
ImageLoader = None  # stub: not used (PIL images passed directly)

__all__ = ['SAM3Model', 'SAM3ImageDetectionOutput', 'SAM3VideoSegmentationOutput']

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SAM3_PATH = os.path.join(_PROJECT_ROOT, 'tools', 'third_party', 'sam3')
if _SAM3_PATH not in sys.path:
    sys.path.insert(0, _SAM3_PATH)


# SAM3ImageDetectionOutput and SAM3VideoSegmentationOutput are imported from
# types.py (lightweight, no torch dep for kernel-side deserialization)


class SAM3Model(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 12.0  # SAM3.1 multiplex: ~3.3GB weights + inference overhead
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 2

    DEVICE = 'cuda'

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()

        if _SAM3_PATH not in sys.path:
            sys.path.insert(0, _SAM3_PATH)

        # Try SAM3.1 (multiplex) first, fall back to SAM3
        checkpoint_31 = os.path.join(
            _PROJECT_ROOT, 'tools', 'third_party', 'sam3', 'weights', 'sam3.1_multiplex.pt'
        )
        checkpoint_30 = os.path.join(
            _PROJECT_ROOT, 'tools', 'third_party', 'sam3', 'weights', 'sam3.pt'
        )
        bpe_path = os.path.join(
            _PROJECT_ROOT, 'tools', 'third_party', 'sam3', 'weights', 'bpe_simple_vocab_16e6.txt.gz'
        )

        if os.path.exists(checkpoint_31):
            print("[SAM3] Loading SAM 3.1 multiplex checkpoint")
            from sam3.model_builder import build_sam3_predictor
            self.predictor = build_sam3_predictor(
                checkpoint_path=checkpoint_31,
                bpe_path=bpe_path if os.path.exists(bpe_path) else None,
                version="sam3.1",
                compile=False,
                warm_up=False,
                use_fa3=False,  # FA3 requires flash-attn-3 package
            )
            self._is_sam31 = True
        elif os.path.exists(checkpoint_30):
            print("[SAM3] Loading SAM 3.0 checkpoint (SAM 3.1 not found)")
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=checkpoint_30,
                bpe_path=bpe_path if os.path.exists(bpe_path) else None,
            )
            self._is_sam31 = False
        else:
            raise FileNotFoundError(
                f'No SAM3 checkpoint found. Looked for:\n'
                f'  SAM 3.1: {checkpoint_31}\n'
                f'  SAM 3.0: {checkpoint_30}\n'
                'Please download following the installation instructions.'
            )

        # SAM3's design: sam3_tracking_predictor.py enters a bfloat16 autocast
        # context in __init__ (self.bf16_context.__enter__()) so that ALL tracker
        # computations run in bfloat16.  However, autocast is thread-local.  Since
        # we call handle_request / handle_stream_request from asyncio.to_thread
        # worker threads, the autocast from __init__ is NOT active during inference.
        # SAM3.1's build_sam3_predictor already enters bf16 autocast in __init__,
        # but the same thread-local issue applies.
        # Fix: store the autocast context and enter it on every worker call via the
        # _bf16_autocast helper below. No model weight modifications needed.

        self.image_loader = image_loader
        self.session_lock = threading.Lock()

    def _extract_frames_to_dir(
        self,
        video_source: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> tuple:
        """Extract video frames to a temp directory as JPEGs for SAM3.

        Args:
            video_source: Path to the video file.
            start_frame: First frame to extract (inclusive). None = 0.
            end_frame: Last frame to extract (exclusive). None = end of video.

        Returns:
            Tuple of (tmp_dir, frame_offset) where frame_offset is the
            absolute index of the first extracted frame (i.e. start_frame or 0).

        Applies ``video_frame_resize_short_edge`` from the global config if set,
        so that SAM3 processes frames at the same resolution as Pi3 / the agent.
        Dense frame extraction (all frames in range) is preserved for propagation quality.
        """
        # Read optional resize setting from config (graceful fallback if unavailable)
        resize_short_edge: Optional[int] = None
        try:
            from workflow.config import get_config
            resize_short_edge = get_config().video_frame_resize_short_edge
        except Exception:
            pass

        tmp_dir = tempfile.mkdtemp(prefix='sam3_frames_')
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {video_source}')

        frame_offset = start_frame or 0

        try:
            # Seek to start_frame if specified
            if start_frame is not None and start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            local_idx = 0
            abs_idx = frame_offset
            while True:
                if end_frame is not None and abs_idx >= end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                if resize_short_edge is not None:
                    w, h = pil_image.size
                    short_side = min(w, h)
                    if short_side > resize_short_edge:
                        scale = resize_short_edge / short_side
                        new_w = int(round(w * scale))
                        new_h = int(round(h * scale))
                        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

                frame_path = os.path.join(tmp_dir, f'{local_idx:06d}.jpg')
                pil_image.save(frame_path, 'JPEG', quality=95)
                local_idx += 1
                abs_idx += 1
        finally:
            cap.release()
        return tmp_dir, frame_offset

    def _save_image_to_dir(self, image: Image.Image) -> str:
        """Save a PIL image to a temp directory as a single-frame video source."""
        tmp_dir = tempfile.mkdtemp(prefix='sam3_img_')
        frame_path = os.path.join(tmp_dir, '000000.jpg')
        image.convert('RGB').save(frame_path, 'JPEG', quality=95)
        return tmp_dir

    def _run_detect(
        self,
        image: Image.Image,
        prompt: str = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
    ) -> SAM3ImageDetectionOutput:
        """Run SAM3 detection on a single image.

        Supports three prompt types (mutually exclusive):
        - **text**: ``prompt`` string (default, detector path)
        - **points**: ``points`` + ``point_labels`` + ``obj_id`` (tracker path, normalized 0-1)
        - **boxes**: ``bounding_boxes`` + ``bounding_box_labels`` (detector path, normalized xywh)

        IMPORTANT: ``close_session`` MUST be in a ``finally`` block.  SAM3's
        predictor caches frame features, grounding state, and tracker memory
        per session.  If ``add_prompt`` or inference raises and the session is
        not closed, that VRAM is leaked permanently, causing OOM under
        concurrent load.  (Unlike Reconstruct/Pi3, which is a stateless
        single forward pass with no session lifecycle.)
        """
        img_dir = self._save_image_to_dir(image)
        orig_w, orig_h = image.size

        # Acquire per-GPU file lock to prevent concurrent inference with
        # co-located actors (e.g. Reconstruct) that share the same physical GPU.
        try:
            with gpu_inference_lock():
                # SAM3's tracker is designed to run under bfloat16 autocast at all times
                # (sam3_tracking_predictor.py __init__ calls self.bf16_context.__enter__()).
                # That context is thread-local and is NOT inherited by asyncio.to_thread
                # worker threads.  Multiple tracker layers (maskmem_backbone.pix_feat_proj,
                # sam_mask_decoder.conv_s0, etc.) receive explicitly-cast bfloat16 tensors
                # and need matching autocast to avoid dtype mismatches.  We re-enter it here.
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                autocast_ctx.__enter__()
                outputs = None
                try:
                    with self.session_lock:
                        result = self.predictor.handle_request({
                            'type': 'start_session',
                            'resource_path': img_dir,
                        })
                        session_id = result['session_id']

                        try:
                            # Build prompt request based on input type.
                            # NOTE: Point prompts use the tracker path which requires
                            # a prior propagate_in_video pass (cached frame outputs).
                            # For single images this cache doesn't exist, so we convert
                            # foreground points to small boxes and use the detector path.
                            # Default output_prob_thresh=0.5 in SAM3 binarizes the
                            # mask after prob threshold, which silently zeroes
                            # out detections whose mask logits don't clear that
                            # bar -- even when the object was actually detected
                            # (out_obj_ids non-empty).  Lower it for image-level
                            # detection so legitimate masks are not lost.
                            prompt_req = {
                                'type': 'add_prompt',
                                'session_id': session_id,
                                'frame_index': 0,
                                'output_prob_thresh': 0.05,
                            }
                            if points is not None:
                                # Convert foreground points to small boxes (detector path)
                                box_size = 0.05  # 5% of image in each direction
                                boxes = []
                                for pt, lbl in zip(points, point_labels):
                                    if lbl == 1:  # foreground only
                                        cx, cy = pt
                                        xmin = max(0.0, cx - box_size)
                                        ymin = max(0.0, cy - box_size)
                                        w = min(box_size * 2, 1.0 - xmin)
                                        h = min(box_size * 2, 1.0 - ymin)
                                        boxes.append([xmin, ymin, w, h])
                                if boxes:
                                    prompt_req['bounding_boxes'] = boxes
                                    prompt_req['bounding_box_labels'] = [1] * len(boxes)
                                else:
                                    prompt_req['text'] = 'object'
                            elif bounding_boxes is not None:
                                prompt_req['bounding_boxes'] = bounding_boxes
                                prompt_req['bounding_box_labels'] = bounding_box_labels
                            else:
                                prompt_req['text'] = prompt

                            prompt_result = self.predictor.handle_request(prompt_req)

                            outputs = prompt_result['outputs']
                        finally:
                            self.predictor.handle_request({
                                'type': 'close_session',
                                'session_id': session_id,
                            })
                finally:
                    autocast_ctx.__exit__(None, None, None)
                    torch.cuda.empty_cache()
        finally:
            # Clean up temp dir outside GPU lock to minimize lock hold time
            import shutil
            shutil.rmtree(img_dir, ignore_errors=True)

        if outputs is None or len(outputs.get('out_obj_ids', [])) == 0:
            return SAM3ImageDetectionOutput(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                masks=np.zeros((0, orig_h, orig_w), dtype=bool),
                labels=[],
            )

        binary_masks = np.array(outputs['out_binary_masks'], dtype=bool)  # (N, H, W)

        # Convert xywh normalized boxes to xyxy pixel
        boxes_xywh = np.array(outputs['out_boxes_xywh'], dtype=np.float32)
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * orig_w
        boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * orig_h
        boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * orig_w
        boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * orig_h

        scores = np.array(outputs['out_probs'], dtype=np.float32)
        n = binary_masks.shape[0]

        return SAM3ImageDetectionOutput(
            boxes=boxes_xyxy,
            scores=scores,
            masks=binary_masks,
            labels=[prompt] * n,
        )

    def _run_segment_video(
        self,
        video_source: str,
        prompts: List[str],
        prompt_frame_idx: int,
        frame_indices: Optional[List[int]],
        points_per_object: Optional[List[List[List[float]]]] = None,
        point_labels_per_object: Optional[List[List[int]]] = None,
        obj_ids: Optional[List[int]] = None,
        boxes_per_object: Optional[List[List[float]]] = None,
        box_labels_per_object: Optional[List[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> SAM3VideoSegmentationOutput:
        """Run SAM3 video segmentation with text, point, or box prompts.

        IMPORTANT: ``close_session`` MUST be in a ``finally`` block — see
        ``_run_detect`` docstring for rationale (session VRAM leak).
        """
        frames_dir, frame_offset = self._extract_frames_to_dir(
            video_source, start_frame=start_frame, end_frame=end_frame,
        )
        total_video_frames = len([
            f for f in os.listdir(frames_dir) if f.endswith('.jpg')
        ])

        all_frame_outputs: Dict[int, dict] = {}

        # Acquire per-GPU file lock to prevent concurrent inference with
        # co-located actors (e.g. Reconstruct) that share the same physical GPU.
        try:
            with gpu_inference_lock():
                # See _run_detect for why we re-enter the bfloat16 autocast here.
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                autocast_ctx.__enter__()
                try:
                    with self.session_lock:
                        result = self.predictor.handle_request({
                            'type': 'start_session',
                            'resource_path': frames_dir,
                        })
                        session_id = result['session_id']

                        # SAM3 multiplex tracker (Sam3MultiplexTrackingWithInteractivity.
                        # add_prompt) routes prompts based on the input type:
                        #
                        #   - text  → super().add_prompt(text_str=...) writes
                        #             text_prompt + per-frame find_inputs into the
                        #             inference_state, so propagate_in_video runs
                        #             VG (visual-grounding) propagation that detects
                        #             the object in every frame.
                        #
                        #   - points → add_sam2_new_points(obj_id, points, labels)
                        #             registers a new object in SAM2 sub-tracker
                        #             memory.  propagate_in_video then runs SAM2
                        #             mask propagation for that object.
                        #
                        #   - boxes → super().add_prompt(boxes_xywh=...) only runs a
                        #             single-frame inference and does NOT register
                        #             the object anywhere.  propagate_in_video
                        #             yields zero frames → silent failure (mask
                        #             returned to client is all zeros).
                        #
                        # Fix: route box prompts through the points path by
                        # converting each box to its center foreground point.
                        # Both points and box requests now share the same
                        # SAM2-registration code path.
                        try:
                            geometric = (points_per_object is not None) or (
                                boxes_per_object is not None
                            )

                            if points_per_object is not None:
                                obj_idx = 0
                                for pts, lbls in zip(
                                    points_per_object, point_labels_per_object
                                ):
                                    # Pick the first foreground point.
                                    fg_pts = [pt for pt, lbl in zip(pts, lbls) if lbl == 1]
                                    if not fg_pts:
                                        continue
                                    self.predictor.handle_request({
                                        'type': 'add_prompt',
                                        'session_id': session_id,
                                        'frame_index': prompt_frame_idx,
                                        'points': [fg_pts[0]],
                                        'point_labels': [1],
                                        'obj_id': obj_idx,
                                    })
                                    obj_idx += 1
                            elif boxes_per_object is not None:
                                # Convert each box (normalized xywh) to its center
                                # point and use the points path so the multiplex
                                # tracker registers the object via add_sam2_new_points.
                                for obj_idx, box in enumerate(boxes_per_object):
                                    xmin, ymin, w, h = box
                                    cx = float(xmin) + float(w) / 2
                                    cy = float(ymin) + float(h) / 2
                                    self.predictor.handle_request({
                                        'type': 'add_prompt',
                                        'session_id': session_id,
                                        'frame_index': prompt_frame_idx,
                                        'points': [[cx, cy]],
                                        'point_labels': [1],
                                        'obj_id': obj_idx,
                                    })
                            else:
                                # Add text prompts for each object
                                for prompt in prompts:
                                    self.predictor.handle_request({
                                        'type': 'add_prompt',
                                        'session_id': session_id,
                                        'frame_index': prompt_frame_idx,
                                        'text': prompt,
                                    })

                            # Propagate through entire video
                            stream_request = {
                                'type': 'propagate_in_video',
                                'session_id': session_id,
                                'propagation_direction': 'both',
                                'start_frame_index': prompt_frame_idx,
                                'max_frame_num_to_track': None,
                            }
                            for item in self.predictor.handle_stream_request(stream_request):
                                fidx = item['frame_index']
                                all_frame_outputs[fidx] = item['outputs']

                            # Safety net: surface silent-fail if a geometric prompt
                            # somehow still registered no object.
                            if geometric and len(all_frame_outputs) == 0:
                                raise RuntimeError(
                                    "SAM3 video tracking returned no frames for the "
                                    "geometric prompt — object failed to register "
                                    "in the video session."
                                )
                        finally:
                            self.predictor.handle_request({
                                'type': 'close_session',
                                'session_id': session_id,
                            })
                finally:
                    autocast_ctx.__exit__(None, None, None)
                    torch.cuda.empty_cache()
        finally:
            # Clean up temp dir outside GPU lock to minimize lock hold time
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

        # Remap local frame indices → absolute by adding frame_offset.
        # SAM3's propagate_in_video returns 0-based local indices;
        # when start_frame is set, local idx 0 = absolute start_frame.
        if frame_offset > 0:
            all_frame_outputs = {
                fidx + frame_offset: out
                for fidx, out in all_frame_outputs.items()
            }

        # Determine which frame indices to return
        if frame_indices is None:
            frame_indices = sorted(all_frame_outputs.keys())

        # Collect masks and scores for the requested frames
        # First pass: find all unique object IDs and max H, W
        all_obj_ids = set()
        sample_h, sample_w = None, None
        for fidx, out in all_frame_outputs.items():
            if out is None:
                continue
            obj_ids = out.get('out_obj_ids', [])
            all_obj_ids.update(obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids))
            masks = out.get('out_binary_masks', None)
            if masks is not None and len(masks) > 0 and sample_h is None:
                sample_h, sample_w = masks.shape[-2], masks.shape[-1]

        all_obj_ids = sorted(all_obj_ids)
        n_obj = len(all_obj_ids)
        obj_id_to_idx = {oid: i for i, oid in enumerate(all_obj_ids)}

        T = len(frame_indices)
        if n_obj == 0 or sample_h is None:
            # No objects detected
            return SAM3VideoSegmentationOutput(
                masks=np.zeros((T, 0, 1, 1), dtype=bool),
                object_ids=[],
                labels=[],
                frame_indices=list(frame_indices),
                num_frames=T,
                _per_frame_scores=np.zeros((T, 0), dtype=np.float32),
            )

        masks_out = np.zeros((T, n_obj, sample_h, sample_w), dtype=bool)
        scores_out = np.zeros((T, n_obj), dtype=np.float32)

        for t, fidx in enumerate(frame_indices):
            out = all_frame_outputs.get(fidx, None)
            if out is None:
                continue
            obj_ids = out.get('out_obj_ids', [])
            obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)
            raw_masks = out.get('out_binary_masks', None)
            raw_scores = out.get('out_probs', None)

            if raw_masks is None or len(raw_masks) == 0:
                continue

            for j, oid in enumerate(obj_ids_list):
                if oid not in obj_id_to_idx:
                    continue
                obj_idx = obj_id_to_idx[oid]
                mask = raw_masks[j]  # (H, W) numpy bool
                masks_out[t, obj_idx] = np.asarray(mask, dtype=bool)
                if raw_scores is not None and j < len(raw_scores):
                    scores_out[t, obj_idx] = float(raw_scores[j])

        # Assign labels: one label per object ID based on prompts
        # Objects are added in order of prompts; if more objects than prompts, reuse last
        labels = []
        for i in range(n_obj):
            prompt_idx = min(i, len(prompts) - 1)
            labels.append(prompts[prompt_idx] if prompts else f'object_{i}')

        return SAM3VideoSegmentationOutput(
            masks=masks_out,
            object_ids=all_obj_ids,
            labels=labels,
            frame_indices=list(frame_indices),
            num_frames=T,
            _per_frame_scores=scores_out,
        )

    @AgentTool.document_output_class(SAM3ImageDetectionOutput)
    async def detect(
        self,
        image_source: str | Image.Image,
        prompt: str = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
    ) -> AgentToolOutput:
        """
        Detects and segments objects in a single image using text, point, or box prompts.
        Args:
            image_source (str | Image.Image): Path to the image file or a PIL Image object.
            prompt (str): Text description of the object to detect (e.g., "dog", "red cup").
            points (List[List[float]]): Normalized 0-1 point coordinates [[x, y], ...] (tracker path).
            point_labels (List[int]): Per-point labels: 1=foreground, 0=background.
            obj_id (int): Object ID for point prompts (required when using points).
            bounding_boxes (List[List[float]]): Normalized xywh boxes [[xmin, ymin, w, h], ...].
            bounding_box_labels (List[int]): Per-box labels (1=foreground).
        """
        if isinstance(image_source, str):
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result
        else:
            image = image_source

        output = await asyncio.to_thread(
            self._run_detect,
            image,
            prompt=prompt,
            points=points,
            point_labels=point_labels,
            obj_id=obj_id,
            bounding_boxes=bounding_boxes,
            bounding_box_labels=bounding_box_labels,
        )
        return self.success(result=output)

    @AgentTool.document_output_class(SAM3VideoSegmentationOutput)
    async def segment_video(
        self,
        video_source: str,
        prompts: List[str] = None,
        prompt_frame_idx: int = 0,
        frame_indices: Optional[List[int]] = None,
        points_per_object: Optional[List[List[List[float]]]] = None,
        point_labels_per_object: Optional[List[List[int]]] = None,
        obj_ids: Optional[List[int]] = None,
        boxes_per_object: Optional[List[List[float]]] = None,
        box_labels_per_object: Optional[List[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> AgentToolOutput:
        """
        Segments and tracks objects across a video using text, point, or box prompts.
        Args:
            video_source (str): Path to the original video file.
            prompts (List[str]): Text descriptions, one per object (detector path).
            prompt_frame_idx (int): Frame index to apply prompts on. Default: 0.
            frame_indices (Optional[List[int]]): Frame indices to return masks for.
            points_per_object (List[List[List[float]]]): Per-object point coords, normalized 0-1.
            point_labels_per_object (List[List[int]]): Per-object point labels (1=fg, 0=bg).
            obj_ids (List[int]): Object IDs for point prompts.
            boxes_per_object (List[List[float]]): Normalized xywh boxes.
            box_labels_per_object (List[int]): Per-box labels.
            start_frame (int): First video frame to process (inclusive). None = 0.
            end_frame (int): Last video frame to process (exclusive). None = end.
        """
        output = await asyncio.to_thread(
            self._run_segment_video,
            video_source,
            prompts or [],
            prompt_frame_idx,
            frame_indices,
            points_per_object=points_per_object,
            point_labels_per_object=point_labels_per_object,
            obj_ids=obj_ids,
            boxes_per_object=boxes_per_object,
            box_labels_per_object=box_labels_per_object,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return self.success(result=output)
