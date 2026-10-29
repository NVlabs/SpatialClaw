"""VLMModule: provides ``vlm.locate()`` and ``vlm.ask_with_thinking()``.

Injected into the kernel as ``vlm``. Each method opens an isolated VLM
session: the VLM sees only the images and the question — no conversation
history, no prior reasoning from the main agent.

- ``vlm.locate(visual_input, question)``        — coordinate grounding,
  short answers, up to 8 images per call.
- ``vlm.ask_with_thinking(visual_input, question)`` — visual reasoning
  with extended thinking budget, up to 64 frames per call.
- ``vlm.point(image, description)``  — ONE point on a described object,
  part or free-space region, returned in pixel coordinates of the image
  (direct pointing instead of box-centre / mask-centroid proxies when the
  answer itself is a point).

Usage in the Jupyter kernel::

    coords = vlm.locate(InputImages[5], "Give the (x, y) center coordinates in 0-1000 normalized scale for the car. Reply with ONLY the numbers.")
    answer = vlm.ask_with_thinking([InputImages[0], InputImages[15], InputImages[30]], "Across these frames, does the person walk toward or away from the camera?")
    pt = vlm.point(InputImages[0], "the handle of the blue mug")   # {"x": 412.3, "y": 288.0, "x_norm": 0.56, "y_norm": 0.52, "raw": "..."} or None
"""

import asyncio
import concurrent.futures
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from spatial_agent.kernel_types.visual_feedback import VisualFeedback


_LOCATE_MAX_IMAGES = 8
_THINKING_MAX_IMAGES = 64


_LOCATE_VERIFY_PREAMBLE = (
    "Before answering, first check whether what the question describes is "
    "clearly visible in the provided image(s). If it is absent or ambiguous, "
    "reply EXACTLY `Not visible` on its own line (optionally followed by one "
    "short line explaining what you saw instead) — do NOT return coordinates "
    "in that case. Only if the requested target is clearly and unambiguously "
    "present, answer the request below.\n\n"
    "Request: "
)


def _parse_point_answer(answer: Any) -> Optional[tuple]:
    """Parse the pointing VLM's answer into normalized (x, y) in [0, 1].

    Accepts ``{"point_2d": [x, y]}`` (fenced or bare JSON, also a one-element
    list of such dicts), ``{"x": .., "y": ..}``, ``[x, y]`` / ``(x, y)`` / ``x y``
    pairs. Values on the 0-1000 grid are divided by 1000; values already in
    [0, 1] are taken as normalized. Returns None for null / "Not visible" /
    unparsable answers.
    """
    if answer is None:
        return None
    s = str(answer).strip()
    if not s or s.startswith("[VLM Error]"):
        return None
    low = s.lower()
    # Explicit refusal: {"point_2d": null} / {'point_2d': None} with any spacing or line breaks, or "Not visible".
    if re.search(r"point(?:_2d)?[\"']?\s*:\s*(?:null|none)\b", low) or low.startswith("not visible"):
        return None
    vals: Optional[List[float]] = None
    json_seen = False
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE) or re.findall(r"(\[\s*\{.*?\}\s*\]|\{[^{}]*\})", s, re.DOTALL)
    for b in blocks:
        try:
            data = json.loads(b.strip())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            json_seen = True
            if ("point_2d" in it or "point" in it) and it.get("point_2d", it.get("point")) is None:
                return None  # an explicit null point is a refusal
            v = it.get("point_2d", it.get("point"))
            try:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    vals = [float(v[0]), float(v[1])]
                    break
                if it.get("x") is not None and it.get("y") is not None:
                    vals = [float(it["x"]), float(it["y"])]
                    break
            except (TypeError, ValueError):
                return None
        if vals:
            break
    if vals is None:
        # A JSON answer without a point key (e.g. Qwen's {"bbox_2d": [...]}) or a long text is NOT a point: never
        # fall back to the first two numbers in it. Only a short bare pair ("412 288", "(412, 288)", "[550, 900]"),
        # optionally followed by one note line, is accepted without JSON.
        if json_seen or len(s) > 300:
            return None
        m = re.match(r"^[\[\(]?\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*[\]\)]?\s*(?:$|\n)", s)
        if not m:
            return None
        vals = [float(m.group(1)), float(m.group(2))]
    x, y = vals
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x, y)
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        return (x / 1000.0, y / 1000.0)
    return None


class VLMModule:
    """Injected into the kernel as ``vlm``.

    The main agent calls ``vlm.locate(...)`` for coordinate grounding and
    ``vlm.ask_with_thinking(...)`` for visual reasoning. Both query
    isolated VLM sessions; the VLM has no memory between calls.
    """

    def __init__(
        self,
        llm_client,  # spatial_agent.llm.client.LLMClient
        locate_system_prompt: str,
        thinking_system_prompt: str,
        session_dir: str,
        session_id: str = "",
        locate_role_params=None,    # LLMRoleParams
        thinking_role_params=None,  # LLMRoleParams
        point_system_prompt: str = "",
    ):
        self._llm_client = llm_client
        self._locate_prompt = locate_system_prompt
        self._thinking_prompt = thinking_system_prompt
        self._point_prompt = point_system_prompt or locate_system_prompt
        self._session_dir = session_dir
        self._session_id = session_id
        self._locate_role_params = locate_role_params
        self._thinking_role_params = thinking_role_params
        self._queries: List[Dict[str, Any]] = []
        self._query_counter = 0
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Public API (called from kernel code)
    # ------------------------------------------------------------------

    def locate(
        self,
        visual_input: Union[VisualFeedback, Image.Image, List],
        question: str,
    ) -> str:
        """Query the grounding VLM. Returns a short text answer (typically coordinates).

        Args:
            visual_input: VisualFeedback, PIL.Image, or list of these (max 8).
            question: Self-contained question, usually requesting
                0-1000 normalized coordinates.

        Returns:
            Text answer from the isolated grounding VLM session. May be
            ``"Not visible"`` (possibly with a short note) when the requested
            target is absent or ambiguous; otherwise the requested coordinates.
        """
        wrapped_question = _LOCATE_VERIFY_PREAMBLE + question
        return self._dispatch(
            visual_input=visual_input,
            question=wrapped_question,
            query_type="locate",
            system_prompt=self._locate_prompt,
            role_params=self._locate_role_params,
            max_images=_LOCATE_MAX_IMAGES,
        )

    def ask_with_thinking(
        self,
        visual_input: Union[VisualFeedback, Image.Image, List],
        question: str,
    ) -> str:
        """Query the thinking VLM. Returns a text answer about the provided frames.

        Args:
            visual_input: VisualFeedback, PIL.Image, or list of these (max 64).
            question: Self-contained question. The session is independent —
                do not embed prior conclusions; phrase the question on its
                own terms so the answer is formed from the images.

        Returns:
            Text answer from the isolated thinking VLM session.
        """
        return self._dispatch(
            visual_input=visual_input,
            question=question,
            query_type="thinking",
            system_prompt=self._thinking_prompt,
            role_params=self._thinking_role_params,
            max_images=_THINKING_MAX_IMAGES,
        )

    def point(
        self,
        image: Union[VisualFeedback, Image.Image, List],
        description: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the pointing VLM for ONE point on a described target.

        Args:
            image: a single VisualFeedback / PIL.Image (a list is accepted; only
                its first image is used, with a printed note).
            description: what to point at — an object, a part of an object, or a
                free-space region, in words (e.g. "the handle of the blue mug",
                "the empty table surface left of the bowl").

        Returns:
            ``{"x": px, "y": py, "x_norm": xn, "y_norm": yn, "raw": answer}`` with
            ``(px, py)`` in PIXEL coordinates of the image you passed and
            ``(xn, yn)`` the same point divided by the image width / height, or
            ``None`` when the target is not visible / ambiguous or the answer
            could not be parsed (the raw answer is printed either way).
        """
        imgs = self._resolve_images(image)
        if len(imgs) > 1:
            print(f"[vlm.point] {len(imgs)} images given; pointing on the first one only.")
        img = imgs[0]
        question = (
            f"Target: {description}\n"
            "Output exactly one point that lies ON the target as JSON on a single line: "
            '{"point_2d": [x, y]} with x and y on a 0-1000 grid of the image '
            "(x from left to right, y from top to bottom). If the target is not visible "
            'or ambiguous, output {"point_2d": null} and one short line saying what you see instead.'
        )
        answer = self._dispatch(
            visual_input=img,
            question=question,
            query_type="point",
            system_prompt=self._point_prompt,
            role_params=self._locate_role_params,
            max_images=1,
            source_desc=(
                str(image)
                if not isinstance(image, list)
                else f"first of a list of {len(image)} images: {image[0]!s}"
            ),
        )
        parsed = _parse_point_answer(answer)
        if parsed is None:
            print(f"[vlm.point] no point for '{description}' (answer: {str(answer)[:120]!r})")
            return None
        xn, yn = parsed
        w, h = img.size
        result = {"x": xn * w, "y": yn * h, "x_norm": xn, "y_norm": yn, "raw": answer}
        print(f"[vlm.point] '{description}' -> pixel ({result['x']:.1f}, {result['y']:.1f}) of a {w}x{h} image")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        visual_input: Union[VisualFeedback, Image.Image, List],
        question: str,
        query_type: str,
        system_prompt: str,
        role_params,
        max_images: int,
        source_desc: Optional[str] = None,
    ) -> str:
        self._query_counter += 1
        query_id = f"vlm_q_{query_type}_{self._query_counter:04d}"

        images = self._resolve_images(visual_input)
        truncated = False
        if len(images) > max_images:
            images = images[:max_images]
            truncated = True

        self._save_query_images(query_id, query_type, images)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                answer = self._executor.submit(
                    asyncio.run,
                    self._llm_client.generate_vision_query(
                        images=images,
                        question=question,
                        system_prompt=system_prompt,
                        role_params=role_params,
                        session_id=self._session_id,
                        usage_session_id=self._session_id,
                    ),
                ).result()
            else:
                answer = loop.run_until_complete(
                    self._llm_client.generate_vision_query(
                        images=images,
                        question=question,
                        system_prompt=system_prompt,
                        role_params=role_params,
                        session_id=self._session_id,
                        usage_session_id=self._session_id,
                    )
                )
        except Exception as exc:
            answer = f"[VLM Error] {exc}"

        if source_desc is None:
            source_desc = (
                str(visual_input)
                if not isinstance(visual_input, list)
                else f"list of {len(visual_input)} images"
            )
        self._queries.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "question": question,
                "answer": answer,
                "num_images": len(images),
                "source": source_desc,
            }
        )

        if truncated:
            print(
                f"[vlm.{query_type}] Showing first {max_images} images; "
                f"remaining were truncated."
            )
        print(f"[VLM Q | {query_type}] {question}")
        print(f"[VLM A | {query_type}] {answer}")

        return answer

    def _resolve_images(
        self, visual_input: Union[VisualFeedback, Image.Image, List]
    ) -> List[Image.Image]:
        import numpy as np
        if isinstance(visual_input, list) or (
            hasattr(visual_input, "__iter__")
            and not isinstance(visual_input, (str, np.ndarray, Image.Image, VisualFeedback))
        ):
            return [self._to_pil(v) for v in visual_input]
        return [self._to_pil(visual_input)]

    @staticmethod
    def _to_pil(obj: Any) -> Image.Image:
        from spatial_agent.kernel_types.frame_image import FrameImage
        import numpy as np

        if isinstance(obj, FrameImage):
            return obj.image
        if isinstance(obj, VisualFeedback):
            return obj.image
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            return Image.fromarray(obj)
        raise TypeError(
            f"Cannot convert {type(obj).__name__} to PIL.Image. "
            f"Accepted: FrameImage, VisualFeedback, PIL.Image, uint8 ndarray."
        )

    def _save_query_images(
        self, query_id: str, query_type: str, images: List[Image.Image]
    ) -> None:
        img_dir = os.path.join(self._session_dir, "vlm_queries", query_type)
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            path = os.path.join(img_dir, f"{query_id}_img_{i}.png")
            img.save(path)

    # ------------------------------------------------------------------
    # Called by feedback_node to retrieve this step's queries
    # ------------------------------------------------------------------

    def get_and_clear_queries(self) -> List[Dict[str, Any]]:
        """Return all queries from the current step and reset the list."""
        queries = self._queries.copy()
        self._queries.clear()
        return queries
