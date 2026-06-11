"""Jupyter kernel lifecycle manager.

Provides a persistent IPython kernel for agent code execution with timeout
enforcement, variable introspection, and restart-with-reinject capability.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import zmq
import zmq.asyncio
from jupyter_client import AsyncKernelManager as JupyterAsyncKM


def _bump_zmq_max_sockets() -> None:
    """Raise the ZMQ per-context socket ceiling.

    jupyter_client's ``start_channels()`` creates ~5 ZMQ sockets per kernel
    session.  The default ``ZMQ_MAX_SOCKETS`` is 1023; if old sockets are not
    fully reclaimed, new channels fail with ``EMFILE``.  Raising the limit
    provides headroom.
    """
    for ctx_cls in (zmq.Context, zmq.asyncio.Context):
        try:
            ctx = ctx_cls.instance()
            ctx.set(zmq.MAX_SOCKETS, 65536)
        except Exception:
            pass


# Apply once at import time.
_bump_zmq_max_sockets()


@dataclass
class ExecutionResult:
    """Structured result from a single cell execution."""

    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None  # None on success; traceback on error
    display_data: List[Any] = field(default_factory=list)
    execution_time_sec: float = 0.0


class JupyterKernelManager:
    """Manages a persistent Jupyter (IPython) kernel for the agent.

    Usage::

        km = JupyterKernelManager(timeout_sec=120)
        kernel_id = await km.start()
        result = await km.execute("print('hello')")
        variables = await km.get_variables()
        await km.shutdown()
    """

    def __init__(
        self,
        timeout_sec: int = 120,
        kernel_name: str = "python3",
    ):
        self.timeout_sec = timeout_sec
        self.kernel_name = kernel_name
        self._km: Optional[JupyterAsyncKM] = None
        self._kc = None  # KernelClient
        self._init_code: str = ""
        self._injection_code: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if a kernel is currently alive and has open channels."""
        return self._km is not None and self._kc is not None

    async def reset_namespace(self) -> None:
        """Clear all user variables and re-run stdlib init code.

        Use between samples to reuse the same kernel process.  Does NOT
        re-run injection code — the caller must inject per-sample objects
        separately.
        """
        if not self.is_running:
            raise RuntimeError("Kernel not running; call start() first.")

        # IPython's %reset clears user namespace (keeps builtins, In/Out)
        result = await self.execute("%reset -f", timeout=10)
        # Errors from %reset are non-fatal (it always works in practice)

        # Clear the ReturnAnswer sentinel from builtins
        await self.clear_sentinel()

        # Re-run stdlib init (numpy, matplotlib, scipy, etc.)
        if self._init_code:
            result = await self.execute(self._init_code, timeout=30)
            if result.error:
                raise RuntimeError(f"Namespace reset init failed: {result.error}")

    async def start(self) -> str:
        """Start a new kernel and return its ID.

        Uses a retry loop with jittered backoff: if ``wait_for_ready`` fails,
        tear everything down, bump the ZMQ context limits, and retry.
        Designed to handle high concurrency (24+ kernels starting at once).
        """
        max_attempts = 5
        ready_timeout = 60  # seconds to wait for kernel readiness
        for attempt in range(1, max_attempts + 1):
            try:
                self._km = JupyterAsyncKM(kernel_name=self.kernel_name)
                await self._km.start_kernel()
                self._kc = self._km.client()
                self._kc.start_channels()
                await self._kc.wait_for_ready(timeout=ready_timeout)
                return str(self._km.kernel_id)
            except Exception as exc:
                # Clean up failed attempt
                try:
                    if self._kc is not None:
                        self._kc.stop_channels()
                except Exception:
                    pass
                try:
                    if self._km is not None:
                        await self._km.shutdown_kernel(now=True)
                except Exception:
                    pass
                self._km = None
                self._kc = None

                if attempt < max_attempts:
                    _bump_zmq_max_sockets()
                    # Jittered exponential backoff to stagger concurrent starts
                    delay = min(2 ** attempt + random.uniform(0, 2), 15)
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Kernel failed to start after {max_attempts} attempts: {exc}"
                    ) from exc

    def set_init_code(self, code: str) -> None:
        """Store initialization code to re-inject after every kernel restart."""
        self._init_code = code

    def set_injection_code(self, code: str) -> None:
        """Store object-injection code to re-inject after every kernel restart."""
        self._injection_code = code

    async def restart(self) -> None:
        """Restart the kernel and re-inject initialization + injection code."""
        if self._km is None:
            raise RuntimeError("Kernel not started")
        await self._km.restart_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        await self._kc.wait_for_ready(timeout=30)
        if self._init_code:
            await self.execute(self._init_code, timeout=30)
        if self._injection_code:
            await self.execute(self._injection_code, timeout=30)

    async def interrupt(self) -> None:
        """Send an interrupt signal to the kernel.

        This sends SIGINT to the kernel process, causing a
        ``KeyboardInterrupt`` in whatever Python code is currently running
        (e.g. ``time.sleep()`` inside a GPU tool wait loop).
        """
        if self._km is not None:
            try:
                await self._km.interrupt_kernel()
            except Exception:
                pass

    async def shutdown(self) -> None:
        """Shut down the kernel."""
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None:
            try:
                await self._km.shutdown_kernel(now=True)
            except Exception:
                pass
        self._km = None
        self._kc = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute *code* in the kernel and return structured result.

        On timeout, the kernel is **not** automatically restarted here --
        that is the caller's responsibility.
        """
        if self._kc is None:
            raise RuntimeError("Kernel not started or has been shut down.")

        timeout = timeout or self.timeout_sec
        t0 = time.monotonic()

        # Send execute request (store_history=True keeps variables alive)
        msg_id = self._kc.execute(code, store_history=True)

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        error_tb: Optional[str] = None
        display_items: List[Any] = []

        try:
            await asyncio.wait_for(
                self._collect_iopub(
                    msg_id, stdout_parts, stderr_parts, display_items
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Interrupt the kernel so blocking code (e.g. GPU server wait
            # loop with time.sleep) gets a KeyboardInterrupt and stops.
            # Without this, the kernel stays busy and all subsequent cells
            # queue behind the still-running code.
            await self.interrupt()
            # Give the kernel a moment to process the interrupt
            await asyncio.sleep(0.5)

            elapsed = time.monotonic() - t0
            return ExecutionResult(
                stdout="".join(stdout_parts),
                stderr="".join(stderr_parts),
                error=f"Cell execution timed out after {elapsed:.1f}s (limit {timeout}s).",
                display_data=display_items,
                execution_time_sec=elapsed,
            )

        # Check for errors captured during collection
        # (set by _collect_iopub via a side-channel)
        if hasattr(self, "_last_error") and self._last_error is not None:
            error_tb = self._last_error
            self._last_error = None

        elapsed = time.monotonic() - t0
        return ExecutionResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            error=error_tb,
            display_data=display_items,
            execution_time_sec=elapsed,
        )

    async def _collect_iopub(
        self,
        msg_id: str,
        stdout_parts: List[str],
        stderr_parts: List[str],
        display_items: List[Any],
    ) -> None:
        """Collect iopub messages until we see the execute_reply idle status."""
        self._last_error = None

        while True:
            try:
                msg = await self._kc.get_iopub_msg(timeout=1.0)
            except Exception:
                await asyncio.sleep(0.05)
                continue

            msg_type = msg.get("msg_type", "")
            content = msg.get("content", {})
            parent = msg.get("parent_header", {})

            # Only process messages for our request
            if parent.get("msg_id") != msg_id:
                continue

            if msg_type == "stream":
                name = content.get("name", "stdout")
                text = content.get("text", "")
                if name == "stdout":
                    stdout_parts.append(text)
                elif name == "stderr":
                    stderr_parts.append(text)

            elif msg_type == "error":
                tb = content.get("traceback", [])
                self._last_error = "\n".join(tb)

            elif msg_type in ("execute_result", "display_data"):
                data = content.get("data", {})
                display_items.append(data)

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

    # ------------------------------------------------------------------
    # Variable introspection
    # ------------------------------------------------------------------

    async def get_variables(self) -> Dict[str, Dict[str, Any]]:
        """Introspect the kernel namespace and return variable metadata.

        Returns a dict: ``var_name -> {type, shape?, dtype?, len?, ...}``.
        Variables starting with ``_`` are excluded.
        """
        introspect_code = r'''
import json as _json_mod
import sys as _sys_mod

_SKIP = {
    '_json_mod', '_sys_mod', '_var_info', '_name', '_val', '_info',
    'In', 'Out', 'get_ipython', 'exit', 'quit',
    'InputImages', 'Metadata', 'tools', 'feedback', 'ReturnAnswer', 'show',
    'np', 'numpy', 'math', 'matplotlib', 'plt', 'scipy',
    'collections', 'itertools', 'functools',
    'ndimage', 'spatial', 'signal', 'optimize',
    'Reconstruct', 'SAM3', 'EasyOCR', 'Graph', 'Time', 'Mask', 'Geometry',
    'Draw', 'RefImages', 'vlm',
}

def _is_injected_name(_n):
    # InputImages_1, InputImages_2, ... in multi-video mode.
    return _n.startswith('InputImages_') and _n[len('InputImages_'):].isdigit()

_var_info = {}
for _name in list(globals().keys()):
    if _name.startswith('_'):
        continue
    if _name in _SKIP or _is_injected_name(_name):
        continue
    _val = globals()[_name]
    if callable(_val) and not hasattr(_val, '__dataclass_fields__'):
        continue
    _info = {"type": type(_val).__name__}
    try:
        if hasattr(_val, 'shape'):
            _info["shape"] = str(_val.shape)
        if hasattr(_val, 'dtype'):
            _info["dtype"] = str(_val.dtype)
        if hasattr(_val, 'frame_indices'):
            _info["frame_indices"] = list(_val.frame_indices)
        if isinstance(_val, (list, tuple)):
            _info["len"] = len(_val)
        if isinstance(_val, dict):
            _info["len"] = len(_val)
            _info["keys"] = list(_val.keys())[:10]
        if hasattr(_val, 'nbytes'):
            _info["size_mb"] = round(_val.nbytes / 1e6, 2)
    except Exception:
        pass
    _var_info[_name] = _info

print(_json_mod.dumps(_var_info))
del _json_mod, _sys_mod, _var_info, _SKIP
'''
        import json

        result = await self.execute(introspect_code, timeout=10)
        if result.error:
            return {}
        try:
            stdout = result.stdout.strip()
            # Take the last line (in case of extra output)
            last_line = stdout.split("\n")[-1]
            return json.loads(last_line)
        except (json.JSONDecodeError, IndexError):
            return {}

    async def check_sentinel(self, sentinel_name: str = "_return_answer_result") -> Optional[Dict]:
        """Check if a sentinel variable exists in the kernel and return its value."""
        import json

        check_code = f"""
import json as _j
try:
    import builtins as _b
    _val = getattr(_b, '{sentinel_name}', None)
    if _val is not None:
        print(_j.dumps(_val))
    else:
        print('__NONE__')
except Exception:
    print('__NONE__')
del _j
"""
        result = await self.execute(check_code, timeout=5)
        if result.error or not result.stdout.strip():
            return None
        text = result.stdout.strip().split("\n")[-1]
        if text == "__NONE__":
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    async def clear_sentinel(self, sentinel_name: str = "_return_answer_result") -> None:
        """Remove the sentinel variable from the kernel."""
        code = f"""
import builtins as _b
if hasattr(_b, '{sentinel_name}'):
    delattr(_b, '{sentinel_name}')
del _b
"""
        await self.execute(code, timeout=5)
