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
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

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
# The cluster's Slurm KillWait is 120 seconds.  Finish the application-level
# drain before that hard scheduler window so unregister/cleanup still has a
# margin and shutdown cannot spend ten minutes holding an ownerless GPU.
_DRAIN_TIMEOUT_SEC = 90

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

class RegistryError(RuntimeError):
    """The shared endpoint registry cannot be read or safely published."""


def _read_registry() -> Dict[str, Any]:
    """Read the registry without converting corruption into an empty object."""
    try:
        with open(_REGISTRY, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return {}

    if raw == "":
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f"Refusing to overwrite corrupt nonempty registry: {_REGISTRY}"
        ) from exc
    if not isinstance(data, dict):
        raise RegistryError(
            f"Refusing to overwrite registry with non-object root: {_REGISTRY}"
        )
    return data


def _atomic_write_registry(data: Dict[str, Any]) -> None:
    """Durably replace the registry using a fsynced same-directory tempfile."""
    if not isinstance(data, dict):
        raise TypeError("Registry root must be a JSON object")

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{_REGISTRY.name}.", suffix=".tmp", dir=_LOGS_DIR,
    )
    try:
        try:
            mode = _REGISTRY.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o600
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, _REGISTRY)
        temporary_path = ""

        directory_fd = os.open(
            _LOGS_DIR, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

@contextlib.contextmanager
def _locked_registry(write=False):
    """Context manager: yields (data, writer).  Call writer(data) to save."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(_REGISTRY_LOCK, "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        data = _read_registry()

        def _write(d):
            _atomic_write_registry(d)

        yield data, _write if write else (lambda _: None)
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


def _register(uid: str, ip: str, http_port: int, tools: list,
              reconstruct_backend: str, num_gpus: int,
              auth_token: str, *,
              health: Optional[Dict[str, Any]] = None,
              synthetic_keepalive_requested: bool = False,
              synthetic_keepalive_effective: bool = False) -> None:
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
            # Additive lifecycle identity. Legacy clients ignore these keys.
            "pool_id": os.environ.get("SPATIAL_AGENT_SERVICE_POOL_ID"),
            "namespace": os.environ.get("SPATIAL_AGENT_SERVICE_NAMESPACE"),
            "slot_id": os.environ.get("SPATIAL_AGENT_SERVICE_SLOT_ID"),
            "generation": os.environ.get("SPATIAL_AGENT_SERVICE_GENERATION"),
            "tool_spec_digest": os.environ.get("SPATIAL_AGENT_TOOL_SPEC_DIGEST"),
            "topology_digest": os.environ.get("SPATIAL_AGENT_TOPOLOGY_DIGEST"),
            "state": "ready",
            "accepting": True,
            "draining": False,
            "service": "spatial-gpu-tools",
            "backend": reconstruct_backend,
            "synthetic_keepalive_requested": bool(
                synthetic_keepalive_requested
            ),
            "synthetic_keepalive_effective": bool(
                synthetic_keepalive_effective
            ),
            "health": dict(health or {}),
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


def _synthetic_keepalive_enabled() -> bool:
    """Preserve legacy availability unless managed lifecycle disables it.

    Managed manager scripts explicitly export ``0`` after sourcing private
    environment files.  An unset variable therefore retains the historical
    unmanaged behavior until the matched lifecycle pilot passes, preventing a
    staged rollout from changing official legacy results through idle reaping.
    """
    raw = os.environ.get("SPATIAL_AGENT_ENABLE_SYNTHETIC_KEEPALIVE")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes"}


def _gpu_identity(num_gpus: int) -> Dict[str, Any]:
    """Return scheduler-visible GPU identity without issuing GPU work."""
    def _devices(variable: str) -> list[str]:
        raw = os.environ.get(variable, "").strip()
        return [value.strip() for value in raw.split(",") if value.strip()]

    return {
        "requested_count": int(num_gpus),
        "cuda_visible_devices": _devices("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": _devices("NVIDIA_VISIBLE_DEVICES"),
        "slurm_job_gpus": _devices("SLURM_JOB_GPUS"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _build_health_identity(
    models: Dict[str, Any],
    expected_tools: list[str],
    reconstruct_backend: str,
    num_gpus: int,
    *,
    synthetic_keepalive_requested: bool,
    synthetic_keepalive_effective: bool,
) -> Dict[str, Any]:
    deployed_tools = [
        tool_name for tool_name in expected_tools
        if _DEPLOYMENT_NAMES.get(tool_name) in models
    ]
    effective_models = {}
    for deployment, model in sorted(models.items()):
        model_identity = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}"
        }
        for attribute in ("MODEL_ID", "DEVICE", "LOCAL_FILES_ONLY"):
            value = getattr(model, attribute, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                model_identity[attribute.lower()] = value
        if hasattr(model, "_is_sam31"):
            model_identity["checkpoint_variant"] = (
                "sam3.1" if bool(model._is_sam31) else "sam3.0"
            )
        effective_models[deployment] = model_identity
    return {
        "status": "ok",
        "service": "spatial-gpu-tools",
        "backend": reconstruct_backend,
        # Dynamic identity lets managed clients detect a stale registry URL
        # that has been rebound by another process or service generation.
        "pool_id": os.environ.get("SPATIAL_AGENT_SERVICE_POOL_ID"),
        "namespace": os.environ.get("SPATIAL_AGENT_SERVICE_NAMESPACE"),
        "slot_id": os.environ.get("SPATIAL_AGENT_SERVICE_SLOT_ID"),
        "generation": os.environ.get("SPATIAL_AGENT_SERVICE_GENERATION"),
        "tool_spec_digest": os.environ.get("SPATIAL_AGENT_TOOL_SPEC_DIGEST"),
        "topology_digest": os.environ.get("SPATIAL_AGENT_TOPOLOGY_DIGEST"),
        "protocol": {
            "name": "spatial-gpu-rpc",
            "version": PROTOCOL_VERSION,
        },
        # Preserve the historical deployment-name list while also exposing
        # configured and effective tool identities explicitly.
        "tools": sorted(models),
        "expected_tools": list(expected_tools),
        "deployed_tools": deployed_tools,
        "effective_models": effective_models,
        "gpu": _gpu_identity(num_gpus),
        "synthetic_keepalive": {
            "requested": bool(synthetic_keepalive_requested),
            "effective": bool(synthetic_keepalive_effective),
        },
    }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _RequestTooLarge(RPCProtocolError):
    pass


class _RuntimeState:
    """Thread-safe reject-new-work and passive real-call attribution."""

    def __init__(self):
        self._lock = threading.Lock()
        self._draining = False
        self._active_requests = 0
        self._request_count = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._next_request_id = 0
        self._active_request_attribution: Dict[int, Dict[str, str]] = {}
        self._last_request_started_at: Optional[str] = None
        self._last_request_finished_at: Optional[str] = None
        self._last_started_request_attribution: Dict[str, str] = {}
        self._last_finished_request_attribution: Dict[str, str] = {}

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def try_begin_request(
        self, attribution: Optional[Dict[str, str]] = None,
    ) -> Optional[int]:
        with self._lock:
            if self._draining:
                return None
            self._next_request_id += 1
            request_id = self._next_request_id
            self._active_requests += 1
            self._request_count += 1
            self._last_request_started_at = self._now()
            request_attribution = dict(attribution or {})
            self._last_started_request_attribution = request_attribution
            self._active_request_attribution[request_id] = request_attribution
            return request_id

    def end_request(
        self, request_id: Optional[int] = None, *, succeeded: bool,
    ) -> None:
        with self._lock:
            if request_id is None and self._active_request_attribution:
                # Compatibility for local state probes predating request IDs.
                request_id = max(self._active_request_attribution)
            attribution = self._active_request_attribution.pop(
                request_id, {},
            )
            self._active_requests = max(0, self._active_requests - 1)
            if succeeded:
                self._completed_requests += 1
            else:
                self._failed_requests += 1
            self._last_request_finished_at = self._now()
            self._last_finished_request_attribution = attribution

    def begin_drain(self) -> None:
        with self._lock:
            self._draining = True

    def snapshot(self, *, include_attribution: bool = True) -> Dict[str, Any]:
        with self._lock:
            snapshot = {
                "state": "draining" if self._draining else "ready",
                "accepting": not self._draining,
                "draining": self._draining,
                "active_requests": self._active_requests,
                "real_request_count": self._request_count,
                "completed_request_count": self._completed_requests,
                "failed_request_count": self._failed_requests,
                "last_real_request_started_at": self._last_request_started_at,
                "last_real_request_finished_at": self._last_request_finished_at,
            }
            if include_attribution:
                snapshot["last_started_request_attribution"] = dict(
                    self._last_started_request_attribution
                )
                snapshot["last_request_attribution"] = dict(
                    self._last_finished_request_attribution
                )
            return snapshot


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


def _create_app(
    models: Dict[str, Any], auth_token: str,
    runtime_state: Optional[_RuntimeState] = None,
    health_identity: Optional[Dict[str, Any]] = None,
):
    """Create the authenticated GPU RPC FastAPI application."""
    from fastapi import FastAPI, Request, Response

    app = FastAPI()
    runtime = runtime_state or _RuntimeState()
    identity = dict(health_identity or {})

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
        # Health is intentionally unauthenticated for registry discovery;
        # expose aggregate activity only, never evaluator ownership metadata.
        snapshot = runtime.snapshot(include_attribution=False)
        return {
            "status": "ok",
            "tools": sorted(models),
            **identity,
            **snapshot,
        }

    def _authorized(request: Request) -> bool:
        authorization = request.headers.get("authorization", "")
        scheme, separator, provided_token = authorization.partition(" ")
        return (
            separator == " "
            and scheme.lower() == "bearer"
            and secrets.compare_digest(
                provided_token.encode("utf-8"), auth_token.encode("utf-8")
            )
        )

    @app.post("/drain")
    async def drain(request: Request):
        if not _authorized(request):
            response = _rpc_response(PermissionError("Authentication required"), 401)
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        runtime.begin_drain()
        return runtime.snapshot(include_attribution=True)

    @app.post("/call")
    async def call_tool(request: Request):
        if not _authorized(request):
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

        attribution = {}
        for key, header in (
            ("owner", "x-spatial-agent-owner"),
            ("cohort", "x-spatial-agent-cohort"),
            ("lease", "x-spatial-agent-lease"),
            ("manifest", "x-spatial-agent-manifest"),
        ):
            value = request.headers.get(header, "").strip()
            if value:
                # Diagnostic identities never enter a tool/model argument.
                # Bound their size so health responses cannot grow without
                # limit even for an authenticated malformed client.
                attribution[key] = value[:256]

        request_id = runtime.try_begin_request(attribution)
        if request_id is None:
            return _rpc_response(
                RuntimeError("GPU server is draining; retry another endpoint"), 503
            )

        succeeded = False
        try:
            method = getattr(model, req["method"])
            if asyncio.iscoroutinefunction(method):
                result = await method(**req.get("kwargs", {}))
            else:
                result = await asyncio.to_thread(method, **req.get("kwargs", {}))
            response = _rpc_response(result)
            succeeded = response.status_code == 200
            return response
        except Exception as exc:
            return _rpc_response(exc, 500)
        finally:
            runtime.end_request(request_id, succeeded=succeeded)

    return app


def _start_http_server(models: Dict[str, Any], port: int, host: str,
                       auth_token: str, runtime_state: _RuntimeState,
                       health_identity: Dict[str, Any]) -> Dict[str, Any]:
    """Start the server and return its verified, non-inference health payload."""
    import uvicorn

    app = _create_app(models, auth_token, runtime_state, health_identity)

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
    url_host = (
        f"[{probe_host}]"
        if ":" in probe_host and not probe_host.startswith("[")
        else probe_host
    )
    health_url = f"http://{url_host}:{port}/health"
    last_error: Optional[Exception] = None
    for _ in range(30):
        time.sleep(0.5)
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=1.0) as response:
                if not 200 <= int(response.status) < 300:
                    raise RuntimeError(
                        f"health endpoint returned HTTP {response.status}"
                    )
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") != "ok":
                raise RuntimeError("health endpoint did not report status=ok")
            if payload.get("expected_tools") != health_identity["expected_tools"]:
                raise RuntimeError("health endpoint tool identity mismatch")
            if payload.get("backend") != health_identity["backend"]:
                raise RuntimeError("health endpoint backend identity mismatch")
            if payload.get("protocol") != health_identity["protocol"]:
                raise RuntimeError("health endpoint protocol identity mismatch")
            if payload.get("gpu") != health_identity["gpu"]:
                raise RuntimeError("health endpoint GPU identity mismatch")
            print(f"[GPU Server] HTTP server listening on {host}:{port}")
            return payload
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"HTTP server did not become healthy on port {port} within 15s: "
        f"{last_error}"
    )


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


def _require_expected_tools(
    expected_tools: list[str], models: Dict[str, Any]
) -> None:
    """Fail startup unless every configured tool deployment is loaded."""
    missing = [
        tool_name for tool_name in expected_tools
        if _DEPLOYMENT_NAMES.get(tool_name) not in models
    ]
    if missing:
        raise RuntimeError(
            "GPU server startup is incomplete; missing expected tool(s): "
            + ", ".join(missing)
        )


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
    runtime_state = _RuntimeState()

    print(f"[GPU Server] Starting (uid={uid}, gpus={args.num_gpus}, "
          f"backend={args.reconstruct_backend}, port={http_port})")

    # Watchdog — SIGALRM fires even when GIL is held.
    signal.signal(signal.SIGALRM, lambda *_: (
        print(f"[GPU Server] ERROR: Startup exceeded {_STARTUP_TIMEOUT_SEC}s", flush=True),
        os._exit(1),
    ))
    signal.alarm(_STARTUP_TIMEOUT_SEC)

    # Synthetic matmul traffic is disabled for normal cluster runs.  Retain a
    # clearly named emergency opt-in for non-production environments only.
    synthetic_keepalive_requested = _synthetic_keepalive_enabled()
    synthetic_keepalive_effective = False
    if synthetic_keepalive_requested:
        from spatial_agent.gpu_models import keepalive as keepalive_module
        keepalive_module.start_gpu_keepalive()
        synthetic_keepalive_effective = bool(
            keepalive_module._keepalive_threads
            and all(
                thread.is_alive()
                for thread in keepalive_module._keepalive_threads
            )
        )
        if not synthetic_keepalive_effective:
            raise RuntimeError(
                "Synthetic keepalive was requested but no worker thread started"
            )
    else:
        print("[Keepalive] Disabled (normal lifecycle mode)", flush=True)

    # Load models
    models = _load_models(tools, args.reconstruct_backend)
    if os.environ.get("SPATIAL_AGENT_SERVICE_POOL_ID", "").strip():
        _require_expected_tools(tools, models)

    health_identity = _build_health_identity(
        models,
        tools,
        args.reconstruct_backend,
        args.num_gpus,
        synthetic_keepalive_requested=synthetic_keepalive_requested,
        synthetic_keepalive_effective=synthetic_keepalive_effective,
    )

    # Start HTTP server and register
    verified_health = _start_http_server(
        models,
        http_port,
        args.host,
        auth_token,
        runtime_state,
        health_identity,
    )
    ip = _advertised_ip(args.host)
    deployed = [t for t in tools if _DEPLOYMENT_NAMES[t] in models]
    _register(
        uid, ip, http_port, deployed, args.reconstruct_backend, args.num_gpus,
        auth_token,
        health=verified_health,
        synthetic_keepalive_requested=synthetic_keepalive_requested,
        synthetic_keepalive_effective=synthetic_keepalive_effective,
    )

    print(f"[GPU Server] READY http://{ip}:{http_port}")
    signal.alarm(0)

    # Block until SIGTERM/SIGINT
    stop = threading.Event()

    def _request_stop(*_):
        # Reject new calls immediately, then let existing calls finish before
        # the process unregisters and exits.
        runtime_state.begin_drain()
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    stop.wait()

    print("[GPU Server] Draining active calls...", flush=True)
    drain_deadline = time.monotonic() + _DRAIN_TIMEOUT_SEC
    while runtime_state.snapshot()["active_requests"] > 0:
        if time.monotonic() >= drain_deadline:
            print("[GPU Server] Drain timeout reached", flush=True)
            break
        time.sleep(1)

    print("[GPU Server] Shutting down...")
    _unregister(uid)
    print("[GPU Server] Done.")


if __name__ == "__main__":
    main()
