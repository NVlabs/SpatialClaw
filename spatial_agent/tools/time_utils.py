"""Time utility module for frame index <-> seconds conversion."""

from spatial_agent.tools.base import CPUTool


class TimeUtils(CPUTool):
    """Frame index and timestamp conversion utilities.

    Usage::

        tools.Time.frame_to_seconds(42)       # -> 1.4 (at 30fps)
        tools.Time.seconds_to_frame(1.4)       # -> 42
        tools.Time.frame_range_to_seconds(0, 30)  # -> 1.0
    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.Time — Frame/Time Conversion (CPU)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `frame_to_seconds` | `(frame_index)` | `float` | Convert frame index to seconds |
| `seconds_to_frame` | `(seconds)` | `int` | Convert seconds to nearest frame (clamped) |
| `frame_range_to_seconds` | `(start_frame, end_frame)` | `float` | Duration between two frames |
| `get_frame_at_time` | `(seconds)` | `int` | Alias for `seconds_to_frame` |

Uses the video's FPS and total frame count (auto-configured from Metadata). Only useful when `Metadata.is_video is True` and `Metadata.fps` is set.
"""

    def __init__(self, fps: float, total_frames: int):
        self.fps = fps
        self.total_frames = total_frames

    def frame_to_seconds(self, frame_index: int) -> float:
        """Convert a frame index to seconds."""
        if not isinstance(frame_index, (int, float)):
            raise TypeError(
                f"`frame_index` must be a number, got {type(frame_index).__name__}."
            )
        if self.fps <= 0:
            return 0.0
        return frame_index / self.fps

    def seconds_to_frame(self, seconds: float) -> int:
        """Convert seconds to the nearest frame index (clamped)."""
        if not isinstance(seconds, (int, float)):
            raise TypeError(
                f"`seconds` must be a number, got {type(seconds).__name__}."
            )
        if self.fps <= 0:
            return 0
        idx = int(round(seconds * self.fps))
        return max(0, min(idx, self.total_frames - 1))

    def frame_range_to_seconds(self, start_frame: int, end_frame: int) -> float:
        """Duration in seconds between two frame indices."""
        if self.fps <= 0:
            return 0.0
        return (end_frame - start_frame) / self.fps

    def get_frame_at_time(self, seconds: float) -> int:
        """Alias for ``seconds_to_frame``."""
        return self.seconds_to_frame(seconds)

    def __repr__(self) -> str:
        return f"TimeUtils(fps={self.fps}, total_frames={self.total_frames})"
