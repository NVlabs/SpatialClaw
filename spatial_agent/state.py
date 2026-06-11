"""LangGraph state definition for the Spatial Understanding Agent.

The agent sees key frames inline in the first message and can call show()
to see additional images. VLM queries use vlm.locate() (grounding) and
vlm.ask_with_thinking() (visual reasoning).
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Verification checklist
# ---------------------------------------------------------------------------

class ChecklistItem(TypedDict):
    """A single verification task tracked across the agent loop."""

    item_id: str  # "chk_001", "chk_002", ...
    description: str  # what to verify
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    status: Literal["PENDING", "VERIFIED", "FLAGGED"]
    added_at_step: int
    resolved_at_step: Optional[int]
    resolution_note: Optional[str]


# ---------------------------------------------------------------------------
# Sub-types stored inside StepResult
# ---------------------------------------------------------------------------

class VLMQuery(TypedDict):
    """A VLM query from vlm.locate() / vlm.ask_with_thinking() and its text-only answer."""

    query_id: str  # unique id, e.g. "vlm_q_locate_0001"
    query_type: str  # "locate", "thinking", or "log"
    image_source: str  # repr of the visual input
    question: str  # the question asked
    answer: Optional[str]  # text answer (None only while in-flight)
    num_images: int  # how many images were sent


class StepResult(TypedDict):
    """Result of a single Jupyter cell execution."""

    step_index: int
    code: str
    stdout: str
    stderr: str
    error: Optional[str]  # None on success; traceback string on failure
    new_variables: Dict[str, str]  # var_name -> summary string
    vlm_queries: List[VLMQuery]  # VLM queries made during this step
    tool_call_count: int  # number of tools.X calls in this cell
    execution_time_sec: float


# ---------------------------------------------------------------------------
# Main agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """LangGraph state for the Spatial Understanding Agent."""

    # --- Identity --------------------------------------------------------
    session_id: str
    sample_id: Optional[str]

    # --- Conversation (text-only, NO images ever) ------------------------
    messages: Annotated[List[AnyMessage], add_messages]

    # --- Loop counters (hard escape conditions) --------------------------
    step_count: int  # current LLM iteration (0-indexed)
    failure_count: int  # CONSECUTIVE failures (reset on success)
    total_tool_calls: int  # cumulative tool invocations across all steps
    total_show_images: int  # cumulative show() images across all steps
    max_steps: int  # default 20
    max_failures: int  # default 5
    max_tool_calls: int  # default 50

    # --- Current step data -----------------------------------------------
    current_llm_response: Optional[Dict[str, str]]
    # parsed JSON: {purpose, reasoning, next_goal, code}
    current_step_result: Optional[StepResult]
    last_error_type: Optional[str]
    # 'llm_call_failed' | 'json_validation_failed' | None

    # --- Kernel state ----------------------------------------------------
    kernel_id: Optional[str]
    # --- Variable registry -----------------------------------------------
    variable_registry: Dict[str, Dict[str, Any]]
    # e.g. {"recon": {"type": "Reconstruction", "num_frames": 3, ...}}

    # --- Final result ----------------------------------------------------
    final_answer: Optional[Dict[str, Any]]
    # set by ReturnAnswer: {text, raw_value}
    last_submitted_answer: Optional[Dict[str, Any]]
    # persists across rejections — never cleared by reflection_node
    termination_reason: Optional[str]
    # 'completed' | 'max_steps' | 'max_failures' | 'max_tool_calls'
    plan: Optional[str]

    # --- Verification checklist (active when enable_reflection=True) ------
    checklist: List[ChecklistItem]
    answer_block_count: int  # consecutive ReturnAnswer blocks (reset on non-block)
    total_answer_attempts: int  # cumulative ReturnAnswer attempts (never resets)

    # --- Input metadata --------------------------------------------------
    instruction: str
    input_metadata: Dict[str, Any]
    # fps, total_frames, is_video, duration_sec, num_images
