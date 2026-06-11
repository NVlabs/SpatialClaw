"""Standalone GPU server for spatial agent tools (Reconstruct, SAM3).

Loads GPU models directly in-process and serves them via HTTP (pickle-serialized).
Agents discover this server via ``logs/gpu_server.json``.

Usage::

    python -m spatial_agent.entrypoints.launch_gpu_server \
        --num_gpus 1 --reconstruct_backend da3
"""

import argparse
import asyncio
import datetime
import contextlib
import fcntl
import importlib
import json
import os
import pickle
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_REGISTRY = _LOGS_DIR / "gpu_server.json"
_REGISTRY_LOCK = _LOGS_DIR / "gpu_server.json.lock"
_STARTUP_TIMEOUT_SEC = 600

# Deployment names — clients reference these in HTTP requests.
_DEPLOYMENT_NAMES = {
    "Reconstruct": "spatial_Reconstruct",
    "SAM3": "spatial_SAM3",
}

# Tool definitions: backend -> {tool_name: (module_path, class_name)}
_GPU_TOOLS = {
    "pi3": {
        "Reconstruct": ("spatial_agent.gpu_models.pi3_model", "Pi3Model"),
        "SAM3": ("spatial_agent.gpu_models.sam3_model", "SAM3Model"),
    },
    "da3": {
        "Reconstruct": ("spatial_agent.gpu_models.da3_model", "DA3Model"),
        "SAM3": ("spatial_agent.gpu_models.sam3_model", "SAM3Model"),
    },
    "mapanything": {
        "Reconstruct": ("spatial_agent.gpu_models.mapanything_model", "MapAnythingModel"),
        "SAM3": ("spatial_agent.gpu_models.sam3_model", "SAM3Model"),
    },
}


# ---------------------------------------------------------------------------
# Registry (gpu_server.json)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _locked_registry(write=False):
    """Context manager: yields (data, writer).  Call writer(data) to save."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(_REGISTRY_LOCK, "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        data = {}
        if _REGISTRY.exists():
            try:
                with open(_REGISTRY) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        def _write(d):
            with open(_REGISTRY, "w") as f:
                json.dump(d, f, indent=2)

        yield data, _write if write else (lambda _: None)
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


def _register(uid: str, ip: str, http_port: int, tools: list,
              reconstruct_backend: str, num_gpus: int) -> None:
    with _locked_registry(write=True) as (data, save):
        data[uid] = {
            "ip": ip,
            "http_port": http_port,
            "tools": tools,
            "reconstruct_backend": reconstruct_backend,
            "num_gpus": num_gpus,
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "create_time": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        }
        save(data)
    print(f"[GPU Server] Registered in {_REGISTRY} (uid={uid})")


def _unregister(uid: str) -> None:
    with _locked_registry(write=True) as (data, save):
        if uid in data:
            del data[uid]
            save(data)
    print(f"[GPU Server] Cleaned up registry entry (uid={uid})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _find_free_port(start: int = 18000, end: int = 19000) -> int:
    import random
    for port in random.sample(range(start, end), end - start):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in {start}-{end}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _start_http_server(models: Dict[str, Any], port: int) -> None:
    """Start a FastAPI server dispatching pickle-serialized calls to models."""
    from fastapi import FastAPI, Request, Response
    import uvicorn

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "tools": list(models.keys())}

    @app.post("/call")
    async def call_tool(request: Request):
        try:
            req = pickle.loads(await request.body())
        except Exception as exc:
            return _pickle_response(exc, 400)

        model = models.get(req.get("deployment"))
        if model is None:
            return _pickle_response(
                RuntimeError(f"Unknown deployment: {req.get('deployment')!r}"), 404)

        try:
            method = getattr(model, req["method"])
            if asyncio.iscoroutinefunction(method):
                result = await method(**req.get("kwargs", {}))
            else:
                result = await asyncio.to_thread(method, **req.get("kwargs", {}))
            return _pickle_response(result)
        except Exception as exc:
            return _pickle_response(exc, 500)

    def _pickle_response(obj, status_code=200):
        try:
            content = pickle.dumps(obj)
        except Exception:
            content = pickle.dumps(RuntimeError(f"{type(obj).__name__}: {obj}"))
            status_code = 500
        return Response(content=content, status_code=status_code,
                        media_type="application/octet-stream")

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port,
                                   log_level="warning", timeout_keep_alive=300),
        daemon=True, name="http-server",
    )
    thread.start()
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                break
        except OSError:
            continue
    else:
        raise RuntimeError(f"HTTP server did not start on port {port} within 15s")
    print(f"[GPU Server] HTTP server listening on 0.0.0.0:{port}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models(tools: list, backend: str) -> Dict[str, Any]:
    """Load GPU models and return {deployment_name: instance}."""
    tool_defs = _GPU_TOOLS.get(backend, _GPU_TOOLS["da3"])
    models: Dict[str, Any] = {}

    for tool_name in tools:
        entry = tool_defs.get(tool_name)
        if not entry:
            print(f"[GPU Server] Warning: Unknown tool {tool_name!r}, skipping.")
            continue

        module_path, class_name = entry
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError) as exc:
            print(f"[GPU Server] Warning: Cannot import {module_path}.{class_name}: {exc}")
            continue

        print(f"[GPU Server] Loading {class_name}...", flush=True)
        models[_DEPLOYMENT_NAMES[tool_name]] = cls(image_loader=None)
        print(f"[GPU Server] {class_name} ready.", flush=True)

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone GPU server")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--reconstruct_backend", type=str, default="da3",
                        choices=["pi3", "da3", "mapanything"])
    parser.add_argument("--http_port", type=int, default=0,
                        help="0 = auto-select")
    args = parser.parse_args()

    uid = uuid.uuid4().hex[:8]
    http_port = args.http_port or _find_free_port()
    tools = ["Reconstruct", "SAM3"]

    print(f"[GPU Server] Starting (uid={uid}, gpus={args.num_gpus}, "
          f"backend={args.reconstruct_backend}, port={http_port})")

    # Watchdog — SIGALRM fires even when GIL is held.
    signal.signal(signal.SIGALRM, lambda *_: (
        print(f"[GPU Server] ERROR: Startup exceeded {_STARTUP_TIMEOUT_SEC}s", flush=True),
        os._exit(1),
    ))
    signal.alarm(_STARTUP_TIMEOUT_SEC)

    # GPU keepalive — prevent DCGM idle-GPU reaper
    from spatial_agent.gpu_models.keepalive import start_gpu_keepalive
    start_gpu_keepalive()

    # Load models
    models = _load_models(tools, args.reconstruct_backend)
    if not models:
        print("[GPU Server] ERROR: No models loaded. Exiting.")
        sys.exit(1)

    # Start HTTP server and register
    _start_http_server(models, http_port)
    ip = _get_local_ip()
    deployed = [t for t in tools if _DEPLOYMENT_NAMES[t] in models]
    _register(uid, ip, http_port, deployed, args.reconstruct_backend, args.num_gpus)

    print(f"[GPU Server] READY http://{ip}:{http_port}")
    signal.alarm(0)

    # Block until SIGTERM/SIGINT
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()

    print("[GPU Server] Shutting down...")
    _unregister(uid)
    print("[GPU Server] Done.")


if __name__ == "__main__":
    main()
