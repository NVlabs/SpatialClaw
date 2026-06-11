"""Main workflow: builds the LangGraph and runs the agent."""

import asyncio
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from spatial_agent.config import SpatialAgentConfig
from spatial_agent.kernel.manager import JupyterKernelManager
from spatial_agent.kernel_types.feedback_module import FeedbackModule
from spatial_agent.kernel_types.input_images import InputImages
from spatial_agent.kernel_types.metadata import Metadata
from spatial_agent.kernel_types.vlm_module import VLMModule
from spatial_agent.llm.client import LLMClient
from spatial_agent.llm.react_prompt import build_react_system_prompt
from spatial_agent.llm.system_prompt import build_system_prompt
from spatial_agent.llm.vision_prompt import (
    get_locate_system_prompt,
    get_thinking_system_prompt,
)
from spatial_agent.logging_utils.logger import SessionLogger
from spatial_agent.nodes.execute_node import execute_node
from spatial_agent.nodes.feedback_node import feedback_node
from spatial_agent.nodes.init_node import init_node
from spatial_agent.nodes.llm_step_node import llm_step_node
from spatial_agent.nodes.plan_node import plan_node
from spatial_agent.nodes.reflection_node import reflection_node
from spatial_agent.nodes.router import force_terminate, should_continue
from spatial_agent.state import AgentState
from spatial_agent.tools import ToolsModule


class KernelPool:
    """Pool of reusable Jupyter kernel managers.

    Kernels are started lazily under a lock (one at a time to avoid
    thundering herd) and returned to the pool after each sample completes.
    The ``init_node`` detects ``km.is_running`` and uses ``reset_namespace()``
    instead of ``start()`` for reused kernels.
    """

    def __init__(self, pool_size: int, timeout_sec: int = 120):
        self._pool_size = pool_size
        self._timeout_sec = timeout_sec
        self._available: asyncio.Queue[JupyterKernelManager] = asyncio.Queue()
        self._start_lock = asyncio.Lock()
        self._total_created = 0

    async def acquire(self) -> JupyterKernelManager:
        """Get a running kernel from the pool, starting a new one if needed.

        New kernels are started under a lock so only one starts at a time,
        preventing system overload from concurrent Jupyter process creation.
        """
        try:
            km = self._available.get_nowait()
            if km.is_running:
                return km
            self._total_created = max(0, self._total_created - 1)
        except asyncio.QueueEmpty:
            pass

        # Serialize kernel creation: starting concurrent Jupyter processes
        # is heavy and tends to overload the host.
        async with self._start_lock:
            try:
                km = self._available.get_nowait()
                if km.is_running:
                    return km
                self._total_created = max(0, self._total_created - 1)
            except asyncio.QueueEmpty:
                pass

            if self._total_created < self._pool_size:
                km = JupyterKernelManager(timeout_sec=self._timeout_sec)
                try:
                    await km.start()
                    self._total_created += 1
                    return km
                except Exception as exc:
                    print(f"[KernelPool] Failed to start kernel: {exc}")
                    raise

        return await self._available.get()

    async def release(self, km: JupyterKernelManager) -> None:
        """Return a kernel to the pool for reuse."""
        if km.is_running:
            await self._available.put(km)
        else:
            self._total_created = max(0, self._total_created - 1)

    async def shutdown_all(self) -> None:
        """Shut down all pooled kernels."""
        while not self._available.empty():
            try:
                km = self._available.get_nowait()
                await km.shutdown()
            except asyncio.QueueEmpty:
                break


class SpatialAgentWorkflow:
    """Orchestrates the spatial understanding agent.

    Usage::

        workflow = SpatialAgentWorkflow(config)
        result = await workflow.arun(
            instruction="...", images=[...], answer="B", session_id="sample_1"
        )
        workflow.shutdown()
    """

    def __init__(self, config: SpatialAgentConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.logger = SessionLogger(config.work_dir or "work_dir/spatial_agent")
        self._kernel_pool = KernelPool(
            pool_size=config.concurrency,
            timeout_sec=config.timeout_sec,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("init_node", init_node)
        graph.add_node("plan_node", plan_node)
        graph.add_node("llm_step_node", llm_step_node)
        graph.add_node("execute_node", execute_node)
        graph.add_node("feedback_node", feedback_node)
        graph.add_node("reflection_node", reflection_node)
        graph.add_node("force_terminate", force_terminate)

        graph.set_entry_point("init_node")
        graph.add_edge("init_node", "plan_node")
        graph.add_edge("plan_node", "llm_step_node")
        graph.add_edge("llm_step_node", "execute_node")
        graph.add_edge("execute_node", "feedback_node")
        graph.add_edge("feedback_node", "reflection_node")
        graph.add_conditional_edges(
            "reflection_node",
            should_continue,
            {
                "llm_step_node": "llm_step_node",
                "force_terminate": "force_terminate",
                END: END,
            },
        )
        graph.add_edge("force_terminate", END)

        return graph.compile()

    async def arun(
        self,
        instruction: str,
        images: List,
        answer: Optional[str] = None,
        session_id: Optional[str] = None,
        frame_indices: Optional[List[int]] = None,
        video_source: Optional[str] = None,
        fps: Optional[float] = None,
        total_video_frames: Optional[int] = None,
        duration_sec: Optional[float] = None,
        image_groups: Optional[List[List]] = None,
        frame_indices_groups: Optional[List[List[int]]] = None,
        fps_per_video: Optional[List[float]] = None,
        total_frames_per_video: Optional[List[int]] = None,
        duration_per_video: Optional[List[float]] = None,
        video_names: Optional[List[str]] = None,
        video_sources_per_video: Optional[List[str]] = None,
        ref_images: Optional[List] = None,
        defer_report: bool = False,
    ) -> Dict[str, Any]:
        """Run the agent on a single sample.

        Args:
            instruction: The question/task.
            images: Concat of all frames (used for single-video / multi-image).
            image_groups: For multi-video samples, per-video frame lists. When
                provided, the kernel receives ``InputImages_1, InputImages_2, ...``
                instead of a single ``InputImages``.
            frame_indices_groups: Per-video absolute frame indices, parallel to
                ``image_groups``.
            fps_per_video / total_frames_per_video / duration_per_video /
                video_names: Per-video metadata exposed via ``Metadata.videos``.

        Returns:
            Dict with 'final_answer', 'termination_reason', etc.
        """
        import shortuuid
        from PIL import Image

        session_id = session_id or shortuuid.uuid()[:8]

        max_edge = self.config.image_max_long_edge
        backend = self.config.reconstruct_backend

        is_multi_video = bool(image_groups) and len(image_groups) > 1
        num_videos = len(image_groups) if is_multi_video else 1

        # Select key frames for main LLM context (eagerly loaded for base64 encoding)
        # `key_frame_list_indices` holds the per-input-images index (i.e. into
        # InputImages or InputImages_<video_idx>); `key_frame_video_idx` is 1-indexed
        # for multi-video and 0 for single-video.
        key_frames = []
        key_frame_indices = []
        key_frame_list_indices = []
        key_frame_video_idx = []
        num_kf = self.config.num_key_frames
        if num_kf > 0:
            if is_multi_video:
                # In multi-video mode num_key_frames is the budget PER VIDEO.
                # Each video contributes up to ``num_kf`` evenly-spaced frames,
                # so a 4-video sample with num_kf=16 yields 64 key frames
                # total. Videos shorter than the budget contribute all frames.
                allocations = [min(num_kf, len(g)) for g in image_groups]
                for vi, (group, alloc) in enumerate(zip(image_groups, allocations)):
                    n = len(group)
                    if n == 0 or alloc <= 0:
                        continue
                    if alloc >= n:
                        local_indices = list(range(n))
                    elif alloc == 1:
                        local_indices = [0]
                    else:
                        local_indices = [round(i * (n - 1) / (alloc - 1)) for i in range(alloc)]
                    fi_group = (frame_indices_groups[vi] if frame_indices_groups
                                and vi < len(frame_indices_groups) else None)
                    for li in local_indices:
                        img = group[li]
                        if isinstance(img, str):
                            img = Image.open(img).convert("RGB")
                        if max_edge:
                            from spatial_agent.gpu_models.image_resize import resize_for_input_images_for_backend
                            img = resize_for_input_images_for_backend(img, max_edge, backend)
                        key_frames.append(img)
                        abs_idx = fi_group[li] if fi_group and li < len(fi_group) else li
                        key_frame_indices.append(abs_idx)
                        key_frame_list_indices.append(li)
                        key_frame_video_idx.append(vi + 1)  # 1-indexed
            elif images:
                n = len(images)
                k = min(num_kf, n)
                if k >= n:
                    selected = list(range(n))
                else:
                    selected = [round(i * (n - 1) / (k - 1)) for i in range(k)]
                for idx in selected:
                    img = images[idx]
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    if max_edge:
                        from spatial_agent.gpu_models.image_resize import resize_for_input_images_for_backend
                        img = resize_for_input_images_for_backend(img, max_edge, backend)
                    key_frames.append(img)
                    abs_idx = frame_indices[idx] if frame_indices and idx < len(frame_indices) else idx
                    key_frame_indices.append(abs_idx)
                    key_frame_list_indices.append(idx)
                    key_frame_video_idx.append(0)

        # Reference images (e.g. MMSI-Video inline <image> tags). Eagerly loaded
        # so they can be base64-encoded into the initial HumanMessage AND
        # injected into the kernel namespace as the `RefImages` list.
        loaded_ref_images: List = []
        if ref_images:
            for p in ref_images:
                img = Image.open(p).convert("RGB") if isinstance(p, str) else p
                if max_edge:
                    from spatial_agent.gpu_models.image_resize import resize_for_input_images_for_backend
                    img = resize_for_input_images_for_backend(img, max_edge, backend)
                loaded_ref_images.append(img)

        # Build InputImages with lazy loading (images load on first access).
        # In multi-video mode we build one InputImages per video; the workflow
        # never instantiates a concatenated InputImages.
        input_images_list: List[InputImages] = []
        if is_multi_video:
            for group, fi in zip(image_groups, frame_indices_groups or [None] * num_videos):
                input_images_list.append(
                    InputImages(group, fi, max_edge=max_edge, backend=backend)
                )
            input_images = None  # No concatenated InputImages in multi-video mode
        else:
            input_images = InputImages(images, frame_indices, max_edge=max_edge, backend=backend)

        is_video = (fps is not None or video_source is not None or is_multi_video)

        if is_multi_video:
            # Per-video metadata for `Metadata.videos` (parallel to InputImages_1, _2, ...).
            videos_meta: List[Dict[str, Any]] = []
            for i, group in enumerate(image_groups):
                vfps = fps_per_video[i] if fps_per_video and i < len(fps_per_video) else None
                vtotal = (total_frames_per_video[i] if total_frames_per_video
                          and i < len(total_frames_per_video) else len(group))
                vdur = (duration_per_video[i] if duration_per_video
                        and i < len(duration_per_video) else None)
                if vdur is None and vfps:
                    vdur = vtotal / vfps
                vname = video_names[i] if video_names and i < len(video_names) else f"video_{i+1}"
                videos_meta.append({
                    "name": vname,
                    "fps": vfps,
                    "num_frames": len(group),
                    "total_frames": vtotal,
                    "duration_sec": vdur,
                })
            metadata = Metadata(
                is_video=True,
                fps=None,
                total_frames=sum(v["num_frames"] for v in videos_meta),
                duration_sec=None,
                num_images=sum(len(g) for g in image_groups),
                video_source=video_source,
                videos=videos_meta,
                num_videos=num_videos,
                reconstruct_max_frames=self.config.reconstruct_max_frames,
            )
        else:
            # Use actual video stats if provided, otherwise derive from loaded images
            actual_total = total_video_frames or len(images)
            actual_duration = duration_sec or ((actual_total / fps) if fps else None)
            metadata = Metadata(
                is_video=is_video,
                fps=fps,
                total_frames=actual_total,
                duration_sec=actual_duration,
                num_images=len(images),
                video_source=video_source,
                videos=None,
                num_videos=1,
                reconstruct_max_frames=self.config.reconstruct_max_frames,
            )

        session_dir = self.logger.get_session_dir(session_id)

        feedback_module = FeedbackModule(
            session_dir=session_dir,
            enable_sighted_feedback=self.config.enable_sighted_feedback,
            session_id=session_id,
        )

        vlm_module = VLMModule(
            llm_client=self.llm_client,
            locate_system_prompt=get_locate_system_prompt(),
            thinking_system_prompt=get_thinking_system_prompt(),
            session_dir=session_dir,
            session_id=session_id,
            locate_role_params=self.config.vlm_grounding_params,
            thinking_role_params=self.config.vlm_params,
        )

        # In multi-video mode SAM3 receives both the per-video InputImages list
        # and the per-video source-file paths so ``segment_video_*(...,
        # video_index=N)`` can pick the right backing video. The single
        # ``input_images`` is the fallback for non-multi-video samples.
        tools_input_images = (
            input_images_list[0] if is_multi_video and input_images_list
            else input_images
        )
        tools_module = ToolsModule(
            config=self.config,
            metadata=metadata,
            input_images=tools_input_images,
            input_images_list=input_images_list if is_multi_video else None,
            video_sources_list=(
                video_sources_per_video if is_multi_video else None
            ),
        )

        # The ReAct baseline restricts output to a single structured tool call
        # per step; single_pass uses the code prompt but should be paired with
        # max_steps=1 and enable_planning=false (set in the dataset config).
        if getattr(self.config, "executor_type", "code") == "react":
            system_prompt = build_react_system_prompt(
                metadata=metadata.to_dict(),
                key_frame_indices=key_frame_indices or None,
                key_frame_list_indices=key_frame_list_indices or None,
                key_frame_video_idx=key_frame_video_idx or None,
                num_ref_images=len(loaded_ref_images),
            )
        else:
            system_prompt = build_system_prompt(
                metadata=metadata.to_dict(),
                key_frame_indices=key_frame_indices or None,
                key_frame_list_indices=key_frame_list_indices or None,
                key_frame_video_idx=key_frame_video_idx or None,
                num_ref_images=len(loaded_ref_images),
            )

        km = await self._kernel_pool.acquire()

        initial_state: AgentState = {
            "session_id": session_id,
            "sample_id": session_id,
            "messages": [],
            "step_count": 0,
            "failure_count": 0,
            "total_tool_calls": 0,
            "total_show_images": 0,
            "max_steps": self.config.max_steps,
            "max_failures": self.config.max_failures,
            "max_tool_calls": self.config.max_tool_calls,
            "current_llm_response": None,
            "current_step_result": None,
            "last_error_type": None,
            "kernel_id": None,
            "variable_registry": {},
            "final_answer": None,
            "termination_reason": None,
            "plan": None,
            "checklist": [],
            "answer_block_count": 0,
            "total_answer_attempts": 0,
            "instruction": instruction,
            "input_metadata": metadata.to_dict(),
        }

        # Each agent step visits ~4-6 LangGraph nodes; the buffer absorbs
        # reflection retries and force-terminate paths.
        run_config = {
            "recursion_limit": self.config.max_steps * 6 + 30,
            "configurable": {
                "kernel_manager": km,
                "llm_client": self.llm_client,
                "tools_module": tools_module,
                "feedback_module": feedback_module,
                "vlm_module": vlm_module,
                "logger": self.logger if self.config.enable_logging else None,
                "system_prompt": system_prompt,
                "agent_config": self.config,
                "input_images": input_images,
                "input_images_list": input_images_list if is_multi_video else None,
                "metadata_obj": metadata,
                "key_frames": key_frames,
                "key_frame_indices": key_frame_indices,
                "key_frame_list_indices": key_frame_list_indices,
                "key_frame_video_idx": key_frame_video_idx,
                "ref_images": loaded_ref_images,
            }
        }

        try:
            final_state = await self.graph.ainvoke(initial_state, config=run_config)
        finally:
            await self._kernel_pool.release(km)

        usage = self.llm_client.pop_session_usage(session_id)

        # When ``defer_report`` is set the caller will regenerate the report
        # with the evaluation score attached.
        report_images = key_frames[:32] + loaded_ref_images
        if self.config.generate_report and not defer_report:
            self.generate_report(
                session_dir, session_id, instruction,
                report_images, answer, final_state,
            )
        if self.config.enable_logging:
            self.logger.log_step(session_id, {
                "event_type": "session_usage",
                **usage,
            })
        if self.config.enable_logging and answer:
            self.logger.log_step(session_id, {
                "event_type": "evaluation",
                "ground_truth": answer,
                "agent_answer": final_state.get("final_answer", {}).get("text", ""),
            })

        return {
            "final_answer": final_state.get("final_answer"),
            "termination_reason": final_state.get("termination_reason"),
            "step_count": final_state.get("step_count", 0),
            "total_tool_calls": final_state.get("total_tool_calls", 0),
            "usage": usage,
            "_report_context": {
                "session_dir": session_dir,
                "instruction": instruction,
                "images": report_images,
                "final_state": {
                    "final_answer": final_state.get("final_answer"),
                    "termination_reason": final_state.get("termination_reason"),
                },
            },
        }

    def generate_report(self, session_dir, session_id, instruction,
                        images, ground_truth, final_state,
                        result_score=None):
        """Generate HTML and PDF reports for a session.

        Args:
            result_score: Optional per-sample score from evaluate_single (0.0-1.0).
        """
        try:
            from spatial_agent.logging_utils.html_report import generate_session_report
            html_path = generate_session_report(
                session_dir=session_dir,
                session_id=session_id,
                instruction=instruction,
                input_images=images,
                ground_truth=ground_truth,
                final_answer=final_state.get("final_answer"),
                termination_reason=final_state.get("termination_reason"),
                result_score=result_score,
            )
            try:
                import weasyprint
                pdf_dir = os.path.join(os.path.dirname(session_dir), "report_pdf")
                os.makedirs(pdf_dir, exist_ok=True)
                weasyprint.HTML(filename=html_path).write_pdf(
                    os.path.join(pdf_dir, f"{session_id}.pdf")
                )
            except ImportError:
                if not getattr(self, "_weasyprint_warned", False):
                    print("[Warning] weasyprint not installed — skipping PDF export. "
                          "Install with: pip install weasyprint")
                    self._weasyprint_warned = True
            except Exception as exc:
                print(f"[Warning] PDF export failed for {session_id}: {exc}")
        except Exception as exc:
            print(f"[Warning] Failed to generate report: {exc}")

    def shutdown(self) -> None:
        """Clean up resources."""
        # Close LLM client pool
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.llm_client.close())
            else:
                loop.run_until_complete(self.llm_client.close())
        except Exception:
            pass

        # Shutdown kernel pool
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._kernel_pool.shutdown_all())
            else:
                loop.run_until_complete(self._kernel_pool.shutdown_all())
        except Exception:
            pass

        # GPU server connections are managed per-kernel by tools/base.py.
        # No cleanup needed here — kernel process exit handles disconnection.
