"""Session logger: trace.jsonl, dialogue.jsonl, and image saving."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from PIL import Image


class SessionLogger:
    """Logs execution traces, dialogue, and images for each session.

    Session directory structure::

        work_dir/session-{session_id}/
            trace.jsonl              # step-by-step events
            vlm_queries/locate/      # images from vlm.locate()
            vlm_queries/thinking/    # images from vlm.ask_with_thinking()
            session_report.html      # interactive HTML report
    """

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def get_session_dir(self, session_id: str) -> str:
        """Create and return the session directory."""
        d = os.path.join(self.work_dir, f"session-{session_id}")
        os.makedirs(d, exist_ok=True)
        return d

    def log_step(self, session_id: str, data: Dict[str, Any]) -> None:
        """Append a step event to trace.jsonl."""
        session_dir = self.get_session_dir(session_id)
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(session_dir, "trace.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def save_image(
        self, session_id: str, image: Image.Image, name: str
    ) -> str:
        """Save an intermediate image.  Returns the file path."""
        session_dir = self.get_session_dir(session_id)
        img_dir = os.path.join(session_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, f"{name}.png")
        image.save(path)
        return path
