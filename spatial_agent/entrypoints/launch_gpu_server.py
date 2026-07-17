"""Standalone GPU server for spatial agent tools (Reconstruct, SAM3).

Loads GPU models directly in-process and serves them via authenticated JSON RPC.
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
import secrets
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from spatial_agent.gpu_rpc import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    RPCProtocolError,
    decode_request,
    encode_error,
    encode_success,
)

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

_ALLOWED_METHODS = {
    "spatial_Reconstruct": frozenset({"reconstruct"}),
    "spatial_SAM3": frozenset({"detect", "segment_video"}),
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
            fd = os.open(_REGISTRY, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(d, f, indent=2)

        yield data, _write if write else (lambda _: None)
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


def _register(uid: str, ip: str, http_port: int, tools: list,
              reconstruct_backend: str, num_gpus: int,
              auth_token: str) -> None:
    with _locked_registry(write=True) as (data, save):
        data[uid] = {
            "ip": ip,
            "http_port": http_port,
            "tools": tools,
            "reconstruct_backend": reconstruct_backend,
            "num_gpus": num_gpus,
            "protocol_version": PROTOCOL_VERSION,
            "auth_token": auth_token,
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


def _find_free_port(host: str, start: int = 18000, end: int = 19000) -> int:
    import random
    try:
        addresses = socket.getaddrinfo(
            host or None,
            0,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except socket.gaierror as exc:
        raise RuntimeError(f"Cannot resolve bind host {host!r}: {exc}") from exc

    for port in random.sample(range(start, end), end - start):
        for family, socktype, proto, _, sockaddr in addresses:
            bind_address = (sockaddr[0], port, *sockaddr[2:])
            try:
                with socket.socket(family, socktype, proto) as sock:
                    sock.bind(bind_address)
                    return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}-{end}")


def _advertised_ip(host: str) -> str:
    """Return the address clients should use for a server bound to ``host``."""
    if host in {"127.0.0.1", "localhost"}:
        return "127.0.0.1"
    if host == "::1":
        return "::1"
    if host in {"", "0.0.0.0", "::"}:
        return _get_local_ip()

    try:
        return socket.getaddrinfo(
            host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )[0][4][0]
    except socket.gaierror as exc:
        raise RuntimeError(f"Cannot resolve advertised host {host!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _RequestTooLarge(RPCProtocolError):
    pass


async def _read_request_body(request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise _RequestTooLarge("Request body is too large")
        except ValueError as exc:
            raise RPCProtocolError("Invalid Content-Length header") from exc

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise _RequestTooLarge("Request body is too large")
    return bytes(body)


def _create_app(models: Dict[str, Any], auth_token: str):
    """Create the authenticated GPU RPC FastAPI application."""
    from fastapi import FastAPI, Request, Response

    app = FastAPI()

    def _rpc_response(obj, status_code=200):
        try:
            content = encode_success(obj) if status_code == 200 else encode_error(obj)
        except Exception as exc:
            content = encode_error(RuntimeError(f"Response serialization failed: {exc}"))
            status_code = 500
        return Response(
            content=content,
            status_code=status_code,
            media_type="application/msgpack",
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "tools": list(models.keys())}

    @app.post("/call")
    async def call_tool(request: Request):
        authorization = request.headers.get("authorization", "")
        scheme, separator, provided_token = authorization.partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(
                provided_token.encode("utf-8"), auth_token.encode("utf-8")
            )
        ):
            response = _rpc_response(PermissionError("Authentication required"), 401)
            response.headers["WWW-Authenticate"] = "Bearer"
            return response

        content_type = (
            request.headers.get("content-type", "")
            .partition(";")[0]
            .strip()
            .lower()
        )
        if content_type != "application/json":
            return _rpc_response(RPCProtocolError("Content-Type must be application/json"), 415)

        try:
            req = decode_request(await _read_request_body(request))
        except _RequestTooLarge as exc:
            return _rpc_response(exc, 413)
        except Exception as exc:
            return _rpc_response(exc, 400)

        model = models.get(req.get("deployment"))
        if model is None:
            return _rpc_response(
                RuntimeError(f"Unknown deployment: {req.get('deployment')!r}"), 404)

        allowed_methods = _ALLOWED_METHODS.get(req["deployment"], frozenset())
        if req["method"] not in allowed_methods:
            return _rpc_response(
                RuntimeError(f"Unknown method for deployment: {req['method']!r}"), 404)

        try:
            method = getattr(model, req["method"])
            if asyncio.iscoroutinefunction(method):
                result = await method(**req.get("kwargs", {}))
            else:
                result = await asyncio.to_thread(method, **req.get("kwargs", {}))
            return _rpc_response(result)
        except Exception as exc:
            return _rpc_response(exc, 500)

    return app


def _start_http_server(models: Dict[str, Any], port: int, host: str,
                       auth_token: str) -> None:
    """Start a FastAPI server dispatching authenticated JSON calls."""
    import uvicorn

    app = _create_app(models, auth_token)

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port,
                                   log_level="warning", timeout_keep_alive=300),
        daemon=True, name="http-server",
    )
    thread.start()
    if host in {"", "0.0.0.0"}:
        probe_host = "127.0.0.1"
    elif host == "::":
        probe_host = "::1"
    else:
        probe_host = host
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection((probe_host, port), timeout=1.0):
                break
        except OSError:
            continue
    else:
        raise RuntimeError(f"HTTP server did not start on port {port} within 15s")
    print(f"[GPU Server] HTTP server listening on {host}:{port}")


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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    uid = uuid.uuid4().hex[:8]
    http_port = args.http_port or _find_free_port(args.host)
    auth_token = secrets.token_urlsafe(32)
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
    _start_http_server(models, http_port, args.host, auth_token)
    ip = _advertised_ip(args.host)
    deployed = [t for t in tools if _DEPLOYMENT_NAMES[t] in models]
    _register(
        uid, ip, http_port, deployed, args.reconstruct_backend, args.num_gpus,
        auth_token,
    )

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
