"""GPU keepalive — prevent cluster idle-GPU job reaper (DCGM_FI_DEV_GPU_UTIL).

One daemon thread per visible GPU runs a short matmul burst every ``interval``
seconds.  The reaper kills jobs with 0% utilization for 90 consecutive
minutes.  Default: 3 s interval, ~150 ms burst = ~5% duty cycle.
"""

import threading
from typing import List

_keepalive_threads: List[threading.Thread] = []
_keepalive_stop = threading.Event()


def start_gpu_keepalive(interval: float = 3.0, size: int = 8192, iters: int = 2) -> None:
    """Spawn one daemon thread per visible GPU that runs periodic matmuls.

    Safe to call multiple times (no-op after the first).
    """
    if _keepalive_threads:
        return

    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return

    for gpu_idx in range(num_gpus):
        def _loop(device=f"cuda:{gpu_idx}"):
            import torch as _torch
            while not _keepalive_stop.wait(timeout=interval):
                try:
                    a = _torch.randn(size, size, device=device, dtype=_torch.float32)
                    b = _torch.randn(size, size, device=device, dtype=_torch.float32)
                    for _ in range(iters):
                        _torch.mm(a, b)
                    _torch.cuda.synchronize(device)
                    del a, b
                except Exception:
                    pass

        t = threading.Thread(target=_loop, daemon=True, name=f"gpu-keepalive-{gpu_idx}")
        t.start()
        _keepalive_threads.append(t)

    print(f"[Keepalive] Started on {num_gpus} GPU(s) (interval={interval}s)")


def stop_gpu_keepalive() -> None:
    """Stop all keepalive threads."""
    _keepalive_stop.set()
    for t in _keepalive_threads:
        t.join(timeout=5.0)
    _keepalive_threads.clear()
    _keepalive_stop.clear()
