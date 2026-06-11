"""Base classes for GPU models.

AgentContext and AgentToolOutput are defined in types.py (single source of truth).
This file provides AgentTool — the base class for GPU model classes.
"""

import inspect
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Optional

from spatial_agent.gpu_models.types import AgentContext, AgentToolOutput


_fifo_cv = threading.Condition()
_fifo_waiters: "deque[int]" = deque()
_fifo_next_ticket: int = 0


@contextmanager
def gpu_inference_lock():
    """Strict FIFO mutex serializing inference on the single GPU owned by this process.

    Every GPU server runs one process per CUDA device (SLURM-allocated, no sharing),
    so an in-process mutex is sufficient — there is no second process contending
    for the same GPU. Callers are woken in the exact order they arrived, preventing
    the starvation pattern that the previous ``fcntl.flock``-based lock allowed
    (Linux flock provides mutual exclusion but no wake-up ordering).
    """
    global _fifo_next_ticket
    ticket = None
    try:
        with _fifo_cv:
            ticket = _fifo_next_ticket
            _fifo_next_ticket += 1
            _fifo_waiters.append(ticket)
            while _fifo_waiters[0] != ticket:
                _fifo_cv.wait()
        yield
    finally:
        # Always remove our ticket — covers both the happy path (head of queue)
        # and any waiter that aborted with an exception inside ``_fifo_cv.wait()``
        # while still queued behind the current holder. ``remove`` is O(n) over a
        # deque bounded by thread-pool size, so cheap.
        if ticket is not None:
            with _fifo_cv:
                try:
                    _fifo_waiters.remove(ticket)
                except ValueError:
                    pass
                _fifo_cv.notify_all()


class AgentTool:
    """Base class for all GPU models."""

    @staticmethod
    def document_output_class(output_class: AgentContext):
        def decorator(method):
            method._output_class = output_class
            return method
        return decorator

    @classmethod
    def get_doc(cls) -> Dict[str, str]:
        docs = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith('_') or name in ['get_doc', 'document_output_class']:
                continue
            method_doc = inspect.getdoc(method) or ''
            if hasattr(method, '_output_class'):
                output_class = method._output_class
                class_name = output_class.__name__
                class_doc = inspect.getdoc(output_class) or ''
                if class_doc:
                    class_doc = class_doc.strip()
                method_doc += f"\n\nReturns:\n    `{class_name}` dataclass, which contains:\n{class_doc}\n"
            if method_doc:
                docs[name] = method_doc
        return docs

    def success(self, result: Any) -> AgentToolOutput:
        return AgentToolOutput(result=result)

    def error(self, msg: str, src: Optional[str] = None) -> AgentToolOutput:
        if src is None:
            try:
                stack = inspect.stack()
                caller_frame = stack[1]
                cls_name = caller_frame[0].f_locals.get('self', '__class__').__class__.__name__
                func_name = caller_frame.function
                src = f'{cls_name}.{func_name}'
            except Exception:
                src = 'Unknown'
        return AgentToolOutput(err_msg=msg, err_src=src)
