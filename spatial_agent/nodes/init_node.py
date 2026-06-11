"""init_node: starts the Jupyter kernel, injects constants, builds the initial messages.

Runs once at the beginning of each agent session. When key frames are
configured, the first user message includes base64-encoded images.
"""

import os
import tempfile
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from spatial_agent.state import AgentState


async def init_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Initialize the kernel and build the first prompt.

    ``config`` is a dict injected via LangGraph's ``configurable`` with keys:
        kernel_manager, llm_client, tools_module, feedback_module, logger,
        system_prompt, agent_config, input_images, metadata_obj, init_injection
    """
    cfg = config["configurable"]
    km = cfg["kernel_manager"]
    tools_module = cfg["tools_module"]
    feedback_module = cfg["feedback_module"]
    vlm_module = cfg["vlm_module"]
    system_prompt = cfg["system_prompt"]
    logger = cfg["logger"]
    input_images = cfg["input_images"]
    input_images_list = cfg.get("input_images_list")
    metadata_obj = cfg["metadata_obj"]

    # 1. Start or reuse the Jupyter kernel
    init_code = _build_init_code()
    if km.is_running:
        if km._init_code:
            # Kernel has been fully initialized before — just reset namespace
            try:
                await km.reset_namespace()
                kernel_id = state.get("kernel_id") or "reused"
            except Exception as exc:
                # Reset failed (timeout, dead kernel, etc.) — full restart
                print(f"[init_node] reset_namespace failed ({exc}), restarting kernel...")
                await km.shutdown()
                kernel_id = await km.start()
                km.set_init_code(init_code)
                result = await km.execute(init_code, timeout=120)
                if result.error:
                    raise RuntimeError(f"Kernel re-init after reset failure: {result.error}")
        else:
            # Kernel started by pool but never initialized — run init code
            kernel_id = str(getattr(km._km, "kernel_id", "pool"))
            km.set_init_code(init_code)
            result = await km.execute(init_code, timeout=120)
            if result.error:
                raise RuntimeError(f"Kernel initialization failed: {result.error}")
    else:
        # No running kernel — start a fresh one
        kernel_id = await km.start()
        km.set_init_code(init_code)
        result = await km.execute(init_code, timeout=120)
        if result.error:
            raise RuntimeError(f"Kernel initialization failed: {result.error}")

    # 2. Inject per-sample objects via cloudpickle temp file
    ref_images = cfg.get("ref_images", []) or []
    injection_code = _build_injection_code(
        feedback_module=feedback_module,
        vlm_module=vlm_module,
        input_images=input_images,
        input_images_list=input_images_list,
        metadata_obj=metadata_obj,
        tools_module=tools_module,
        ref_images=ref_images,
    )
    km.set_injection_code(injection_code)
    result = await km.execute(injection_code, timeout=300)
    if result.error:
        raise RuntimeError(f"Object injection failed: {result.error}")

    # 4. Build initial messages (multimodal if key frames available)
    key_frames = cfg.get("key_frames", [])
    key_frame_indices = cfg.get("key_frame_indices", [])
    key_frame_list_indices = cfg.get("key_frame_list_indices", [])
    key_frame_video_idx = cfg.get("key_frame_video_idx") or []
    is_multi_video = bool(input_images_list)
    if is_multi_video:
        num_total_images = sum(len(ii) for ii in input_images_list)
    else:
        num_total_images = len(cfg.get("input_images") or [])

    def _kf_var(pos: int, li: int, ai: int) -> str:
        """Render a single key frame mapping line (multi-video aware)."""
        if is_multi_video and pos < len(key_frame_video_idx) and key_frame_video_idx[pos]:
            v = key_frame_video_idx[pos]
            return f"  #{pos+1} → InputImages_{v}[{li}] (video frame {ai})"
        return f"  #{pos+1} → InputImages[{li}] (video frame {ai})"

    if key_frames or ref_images:
        from spatial_agent.llm.client import image_to_base64_url

        content_parts = []

        if key_frames:
            # Build compact mapping text
            mapping_lines = []
            n_kf = len(key_frames)
            if key_frame_list_indices:
                if n_kf <= 16:
                    for i, (li, ai) in enumerate(zip(key_frame_list_indices, key_frame_indices)):
                        mapping_lines.append(_kf_var(i, li, ai))
                else:
                    for i in range(5):
                        li, ai = key_frame_list_indices[i], key_frame_indices[i]
                        mapping_lines.append(_kf_var(i, li, ai))
                    mapping_lines.append(f"  ... ({n_kf - 10} more) ...")
                    for i in range(n_kf - 5, n_kf):
                        li, ai = key_frame_list_indices[i], key_frame_indices[i]
                        mapping_lines.append(_kf_var(i, li, ai))
                mapping_text = "\n".join(mapping_lines)
            else:
                mapping_text = ""

            input_images_label = (
                f"InputImages_1..{len(input_images_list)} (across {len(input_images_list)} videos)"
                if is_multi_video else "InputImages"
            )
            header = (
                f"Here are {n_kf} key frames — a visual overview subset of "
                f"{input_images_label} ({num_total_images} total frames)."
            )
            if mapping_text:
                header += f"\nKey frame mapping (context position → variable[index] → video frame):\n{mapping_text}"

            content_parts.append({
                "type": "text",
                "text": header,
            })
            for img in key_frames:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img)},
                })

        if ref_images:
            content_parts.append({
                "type": "text",
                "text": (
                    f"Reference images for inline `[reference image #N]` tags "
                    f"in the question ({len(ref_images)} total, 1-indexed). "
                    f"In your Python code, access them as `RefImages[N-1]`."
                ),
            })
            for img in ref_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img)},
                })

        content_parts.append({"type": "text", "text": state["instruction"]})
        user_message = HumanMessage(content=content_parts)
    else:
        user_message = HumanMessage(content=state["instruction"])

    messages = [
        SystemMessage(content=system_prompt),
        user_message,
    ]

    # 5. Snapshot variable registry after injection so the first feedback diff
    #    only shows variables created by the LLM, not pre-injected modules/tools.
    initial_vars = await km.get_variables()

    # 6. Log initialization
    if logger:
        logger.log_step(state["session_id"], {
            "event_type": "init",
            "kernel_id": kernel_id,
            "instruction": state["instruction"],
            "system_prompt": system_prompt,
        })

    return {
        "messages": messages,
        "kernel_id": kernel_id,
        "step_count": 0,
        "failure_count": 0,
        "total_tool_calls": 0,
        "total_show_images": 0,
        "variable_registry": initial_vars,
        "current_llm_response": None,
        "current_step_result": None,
        "final_answer": None,
        "last_submitted_answer": None,
        "termination_reason": None,
    }


def _build_init_code() -> str:
    """Build the Python code that initializes the kernel namespace with stdlib.

    Includes pre-imports of modules that cloudpickle deserialization will need
    (Phase 2 of the injection code).  Kernel starts are serialized by
    KernelPool's start lock, so these imports are naturally staggered across
    processes — warming the OS page cache and avoiding Lustre I/O storms when
    many kernels cold-start simultaneously.
    """
    return """
# === Spatial Agent Kernel Initialization ===
import numpy as np
import math
import collections
import itertools
import functools

# Matplotlib non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# scipy
try:
    import scipy
    from scipy import ndimage, spatial, signal, optimize
except ImportError:
    pass

# Pre-import modules needed by cloudpickle deserialization (injection Phase 2).
# Without this, 16+ concurrent kernels all try to import from Lustre at once,
# causing filesystem I/O storms and 300s+ timeouts.
try:
    import cloudpickle
    import requests
    import spatial_agent.gpu_models
    import spatial_agent.tools.base
    import spatial_agent.tools.sam3_tool
    import spatial_agent.tools.reconstruct_tool
    import spatial_agent.tools.geometry_utils
    import spatial_agent.kernel_types.frame_image
    import spatial_agent.kernel_types.return_answer
    import spatial_agent.kernel_types.per_frame_types
    import spatial_agent.kernel_types.visual_feedback
    import spatial_agent.kernel_types.input_images
    import spatial_agent.kernel_types.feedback_module
    import spatial_agent.kernel_types.vlm_module
except ImportError:
    pass

print("[Kernel] stdlib init complete.")
"""


def _build_show_injection_code() -> str:
    """Build code to inject show() and plt.show() interception into the kernel."""
    return """
show = feedback.show

# Monkey-patch plt.show() to capture figures as images
import matplotlib.pyplot as _plt_module
_original_plt_show = _plt_module.show

def _patched_plt_show(*args, **kwargs):
    import io as _io
    from PIL import Image as _PILImage
    try:
        fig_nums = _plt_module.get_fignums()
        if not fig_nums:
            print("[plt.show patch] No open figures.")
            return
        figs = [_plt_module.figure(n) for n in fig_nums]
        for fig in figs:
            # Extract title for label
            axes = fig.get_axes()
            title = ""
            for ax in axes:
                t = ax.get_title()
                if t:
                    title = t
                    break
            if not title:
                title = fig.get_suptitle() if hasattr(fig, 'get_suptitle') else ""
            label = title if title else "matplotlib figure"
            # Render to PIL
            buf = _io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = _PILImage.open(buf).copy()
            buf.close()
            feedback.show(img, label=label)
        _plt_module.close('all')
    except Exception as _e:
        print(f"[plt.show patch] Error: {_e}")

_plt_module.show = _patched_plt_show
print("[Kernel] show() and plt.show() interception enabled.")
"""


def _build_injection_code(
    feedback_module,
    vlm_module,
    input_images,
    input_images_list,
    metadata_obj,
    tools_module,
    ref_images=None,
) -> str:
    """Serialize objects to a temp file and return code to load them in the kernel.

    The injection is split into three phases to avoid a subtle bug where
    importing GPU model modules shadows the ``tools`` variable:

    Phase 1 — Import ``spatial_agent.gpu_models`` FIRST (before deserializing
              ToolsModule) to populate ``sys.modules`` entries that
              cloudpickle needs for deserializing GPU tool output types
              (e.g. ``Pi3ReconstructionOutput``). Also imports legacy
              ``tools.apis`` for backward compatibility with old pickles.

    Phase 2 — Deserialize the ToolsModule and other per-sample objects
              from a temp pickle file.

    Phase 3 — Create a proper ``types.ModuleType`` wrapper around
              ToolsModule and install it in ``sys.modules["tools"]``.
              This makes *every* import style the LLM might use work:
              ``tools.Reconstruct``, ``from tools import Reconstruct``, ``import tools``,
              or just ``Pi3`` directly (top-level injection).
    """
    import cloudpickle

    from spatial_agent.kernel_types.return_answer import ReturnAnswer
    from spatial_agent.tools import ToolsModule

    is_multi_video = bool(input_images_list)
    objects = {
        "feedback": feedback_module,
        "vlm": vlm_module,
        "Metadata": metadata_obj,
        "tools": tools_module,
        "ReturnAnswer": ReturnAnswer,
    }
    if is_multi_video:
        # Inject one InputImages_<N> per video (1-indexed). The flat
        # ``InputImages`` is intentionally NOT created in multi-video mode.
        for i, ii in enumerate(input_images_list, start=1):
            objects[f"InputImages_{i}"] = ii
    else:
        objects["InputImages"] = input_images
    if ref_images:
        objects["RefImages"] = list(ref_images)

    # Write to a temp file the kernel can read
    fd, pkl_path = tempfile.mkstemp(prefix="spatial_agent_inject_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        cloudpickle.dump(objects, f)

    safe_path = pkl_path.replace("\\", "\\\\")
    tool_names = list(ToolsModule.TOOL_NAMES)

    # Conditional lines — empty string when no ref images, preserving the
    # pre-feature kernel namespace byte-for-byte.
    refimages_load = 'RefImages = _injected["RefImages"]\n' if ref_images else ''
    refimages_log = ", RefImages" if ref_images else ""

    if is_multi_video:
        n_videos = len(input_images_list)
        input_images_load = "\n".join(
            f'InputImages_{i} = _injected["InputImages_{i}"]'
            for i in range(1, n_videos + 1)
        )
        input_images_log_names = ", ".join(
            f"InputImages_{i}" for i in range(1, n_videos + 1)
        )
    else:
        input_images_load = 'InputImages  = _injected["InputImages"]'
        input_images_log_names = "InputImages"

    return f"""
import time as _time
_t0 = _time.monotonic()
def _elapsed():
    return f"{{_time.monotonic() - _t0:.2f}}s"

# === Phase 1: Pre-import output types for deserialization ===
import sys as _sys
try:
    import spatial_agent.gpu_models as _gpu_models  # lightweight types only, no torch
except Exception:
    _gpu_models = None
print(f"[Kernel inject] Phase 1 (imports): {{_elapsed()}}")

# === Phase 2: Deserialize per-sample objects ===
import cloudpickle as _cp
with open("{safe_path}", "rb") as _f:
    _injected = _cp.load(_f)

feedback     = _injected["feedback"]
vlm          = _injected["vlm"]
{input_images_load}
Metadata     = _injected["Metadata"]
ReturnAnswer = _injected["ReturnAnswer"]
_tm          = _injected["tools"]      # ToolsModule instance (DO NOT name this 'tools')
{refimages_load}print(f"[Kernel inject] Phase 2 (deserialize): {{_elapsed()}}")

# === Phase 3: Create a real module and install in sys.modules ===
import types as _types

_tools_mod = _types.ModuleType("tools")
_tools_mod.__package__ = "tools"
_tools_mod.__path__ = []

# Copy every tool attribute from ToolsModule onto the module AND register
# each tool as a submodule in sys.modules so that ALL import styles work:
#   tools.Reconstruct.Reconstruct(...)          — attribute access
#   from tools import Reconstruct               — module import
#   from tools.Reconstruct import Reconstruct   — submodule import (needs sys.modules entry)
#   Reconstruct.Reconstruct(...)        — top-level (injected below)
for _name in {tool_names!r}:
    if hasattr(_tm, _name):
        _tool_obj = getattr(_tm, _name)
        setattr(_tools_mod, _name, _tool_obj)
        # Register as a fake submodule so 'from tools.Reconstruct import X' works
        _sub = _types.ModuleType(f"tools.{{_name}}")
        _sub.__package__ = "tools"
        for _attr in dir(_tool_obj):
            if not _attr.startswith("_"):
                setattr(_sub, _attr, getattr(_tool_obj, _attr))
        _sys.modules[f"tools.{{_name}}"] = _sub

# Install — now 'import tools' and 'from tools import X' resolve here
_sys.modules["tools"] = _tools_mod

# Bind in kernel namespace
tools = _tools_mod

# Also inject each tool at TOP LEVEL so the LLM can use Reconstruct.Reconstruct()
# directly, without the 'tools.' prefix. This survives even if the LLM
# accidentally reassigns 'tools'.
{chr(10).join(f"{n} = getattr(_tm, {n!r}, None)" for n in tool_names)}

# Register 'feedback' as a fake module so 'import feedback' works too
_fb_mod = _types.ModuleType("feedback")
for _attr in dir(feedback):
    if not _attr.startswith("_"):
        setattr(_fb_mod, _attr, getattr(feedback, _attr))
_sys.modules["feedback"] = _fb_mod

# Same for 'vlm'
_vlm_mod = _types.ModuleType("vlm")
for _attr in dir(vlm):
    if not _attr.startswith("_"):
        setattr(_vlm_mod, _attr, getattr(vlm, _attr))
_sys.modules["vlm"] = _vlm_mod

print(f"[Kernel inject] Phase 3 (module wiring): {{_elapsed()}}")

del _cp, _f, _injected, _tm, _types, _tools_mod, _name, _tool_obj, _sub, _fb_mod, _vlm_mod, _attr

print(f"[Kernel] Injected in {{_elapsed()}}: feedback, vlm, {input_images_log_names}, Metadata, tools, ReturnAnswer{refimages_log}")
print("[Kernel] Top-level tools: {', '.join(tool_names)}")

# === Phase 4: Inject show() function and plt.show() interception ===
{_build_show_injection_code()}
"""
