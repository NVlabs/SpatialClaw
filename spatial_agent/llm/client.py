"""LLM client with vLLM load-balanced discovery from logs/serve.json.

Features:
- Client pooling: one AsyncOpenAI per endpoint, reused across requests
- Periodic re-discovery: re-reads serve.json every 60s (not just on failure)
- Health checking: startup probe + runtime failure tracking
- Least-connections routing: picks endpoint with fewest in-flight requests
- Sticky routing: hash(session_id) → endpoint for prefix cache hits (load-aware)
"""

import asyncio
import base64
import threading
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
import random
import re
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Mapping, Optional, Tuple

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    Timeout,
)
from PIL import Image

from spatial_agent.config import LLMRoleParams, SpatialAgentConfig
from spatial_agent.llm.response_schema import LLMResponseValidator

logger = logging.getLogger(__name__)

# Server-wait configuration
_SERVER_WAIT_TIMEOUT_SEC = 4 * 3600  # 4 hours
_SERVER_WAIT_POLL_SEC = 30  # poll every 30 seconds

# Re-discovery TTL
_DISCOVERY_TTL_SEC = 60.0

# Unhealthy endpoint cooldown
_UNHEALTHY_COOLDOWN_SEC = 60.0

# Startup health check timeout per endpoint
_HEALTH_CHECK_TIMEOUT_SEC = 3

_RESERVED_LIFECYCLE_HEADERS = frozenset({
    "x-spatial-agent-owner",
    "x-spatial-agent-cohort",
    "x-spatial-agent-lease",
    "x-spatial-agent-manifest",
})
_FROZEN_HF_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


_COMMON_LIFECYCLE_ENV_KEYS = (
    "SPATIAL_AGENT_LEASE_ID",
    "SPATIAL_AGENT_COHORT_ID",
    "SPATIAL_AGENT_OWNER_ID",
    "SPATIAL_AGENT_WORKLOAD_POLICY_DIGEST",
    "SPATIAL_AGENT_SERVICE_MANIFEST_DIGEST",
)
_VLLM_LIFECYCLE_ENV_KEYS = _COMMON_LIFECYCLE_ENV_KEYS + (
    "SPATIAL_AGENT_VLLM_POOL_ID",
    "SPATIAL_AGENT_VLLM_NAMESPACE",
    "SPATIAL_AGENT_VLLM_SERVER_SPEC_DIGEST",
    "SPATIAL_AGENT_VLLM_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST",
)
_TOOL_LIFECYCLE_ENV_KEYS = (
    "SPATIAL_AGENT_TOOL_POOL_ID",
    "SPATIAL_AGENT_TOOL_NAMESPACE",
    "SPATIAL_AGENT_TOOL_SPEC_DIGEST",
    "SPATIAL_AGENT_TOOL_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST",
)
_GLOBAL_LIFECYCLE_ENV_KEYS = (
    _VLLM_LIFECYCLE_ENV_KEYS + _TOOL_LIFECYCLE_ENV_KEYS
)


def _vllm_lifecycle_requested() -> bool:
    """Return whether any core lifecycle-envelope key was supplied.

    Presence, rather than truthiness, is intentional: an empty or malformed
    value -- including residue for the other service type -- must fail managed
    discovery closed, not silently select legacy/unscoped capacity.
    """
    return any(name in os.environ for name in _GLOBAL_LIFECYCLE_ENV_KEYS)


def _managed_vllm_envelope() -> Optional[Dict[str, Any]]:
    """Validate and return the exact managed-client vLLM envelope."""
    values = {
        name: os.getenv(name, "").strip()
        for name in _VLLM_LIFECYCLE_ENV_KEYS
        if name != "SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST"
    }
    missing = sorted(name for name, value in values.items() if not value)
    raw_slots = os.getenv("SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST", "")
    try:
        decoded_slots = json.loads(raw_slots)
    except (json.JSONDecodeError, TypeError):
        decoded_slots = None
    if (
        not isinstance(decoded_slots, list)
        or not decoded_slots
        or any(not isinstance(slot, str) or not slot.strip() for slot in decoded_slots)
    ):
        missing.append("SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST")
        slots: set[str] = set()
    else:
        slots = {slot.strip() for slot in decoded_slots}
        if len(slots) != len(decoded_slots):
            missing.append("SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST")

    if missing:
        logger.error(
            "Managed vLLM discovery has an incomplete or malformed exact "
            "client envelope (%s); failing closed",
            ", ".join(sorted(set(missing))),
        )
        return None
    return {**values, "slots": slots}


class LLMGenerationTimeoutError(TimeoutError):
    """An inference request exceeded its explicit outer generation deadline."""

    def __init__(self, request_kind: str, attempts: int):
        self.request_kind = request_kind
        self.attempts = attempts
        super().__init__(
            f"{request_kind} generation timed out after {attempts} attempt(s)"
        )


class _TransportAsyncTimeoutError(asyncio.TimeoutError):
    """An asyncio timeout raised by the transport, not by our outer deadline."""


def _positive_timeout_override_from_env(name: str) -> Optional[float]:
    """Return a positive timeout override, or ``None`` when unset/invalid."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return None
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, raw)
        return None
    return value


def resolve_request_timeouts(config: SpatialAgentConfig) -> Dict[str, Any]:
    """Resolve effective request budgets and their provenance.

    Explicit config values win absolutely. This is important for scoped
    benchmark policies: inherited timeout environment variables must not
    silently alter the policy's accuracy or tail behavior. When a value is not
    pinned, the historical environment/default resolution is preserved.
    """

    def _resolve(
        configured: Optional[float],
        env_name: str,
        default: float,
    ) -> Tuple[float, str]:
        if configured is not None:
            return float(configured), "config"
        env_value = _positive_timeout_override_from_env(env_name)
        if env_value is not None:
            return env_value, f"env:{env_name}"
        return float(default), "default"

    llm_seconds, llm_source = _resolve(
        getattr(config, "llm_request_timeout_sec", None),
        "SPATIAL_AGENT_LLM_REQUEST_TIMEOUT_SEC",
        600.0,
    )
    vlm_seconds, vlm_source = _resolve(
        getattr(config, "vlm_request_timeout_sec", None),
        "SPATIAL_AGENT_VLM_REQUEST_TIMEOUT_SEC",
        float(config.vlm_query_timeout_sec),
    )
    return {
        "llm_request_timeout_sec": llm_seconds,
        "llm_request_timeout_source": llm_source,
        "vlm_request_timeout_sec": vlm_seconds,
        "vlm_request_timeout_source": vlm_source,
    }


def _sdk_transport_timeout_for_deadline(deadline_sec: float) -> Timeout:
    """Keep SDK transport phases distinct from the outer generation budget.

    Read timeout is deliberately later than our outer deadline. Connect, pool,
    and write phases retain shorter deadlines, so their SDK ``APITimeoutError``
    remains a retryable transport failure rather than masquerading as a
    completed generation tail.
    """
    deadline = max(float(deadline_sec), 0.001)
    return Timeout(
        connect=min(5.0, max(0.001, deadline / 4.0)),
        pool=min(30.0, max(0.001, deadline / 4.0)),
        write=min(600.0, max(0.001, deadline / 2.0)),
        read=deadline + max(1.0, deadline * 0.01),
    )


async def _with_transport_timeout_identity(awaitable):
    """Prevent an inner asyncio timeout from being confused with wait_for()."""
    try:
        return await awaitable
    except asyncio.TimeoutError as exc:
        raise _TransportAsyncTimeoutError(str(exc)) from exc


def image_to_base64_url(img: Image.Image, max_size: int = 1024) -> str:
    """Encode a PIL image as a base64 data URI, resizing if needed."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _images_to_base64_urls(images: List[Image.Image]) -> List[str]:
    """Encode one request's images in order using the historical encoder."""
    return [image_to_base64_url(img) for img in images]


def _exception_type_chain(exc: BaseException) -> str:
    """Return bounded exception/cause type names without logging payload data."""
    names: List[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen and len(names) < 8:
        seen.add(id(current))
        names.append(type(current).__name__)
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
    return " -> ".join(names)


def _retry_request_id_fail_open() -> Optional[str]:
    """Create a logical-request identity without affecting generation."""
    try:
        return uuid.uuid4().hex
    except Exception:
        return None


def _retry_payload_sha256(
    model: str,
    messages: List[Dict[str, Any]],
    api_kwargs: Mapping[str, Any],
) -> Optional[str]:
    """Digest the OpenAI JSON request body without transport metadata.

    The SDK merges ``extra_body`` into the top-level JSON object. ``timeout``,
    ``extra_headers``, and ``extra_query`` are request controls rather than
    body fields. Excluding them also prevents header or query credentials from
    contributing to the audit digest.
    """
    try:
        request_body: Dict[str, Any] = {"model": model, "messages": messages}
        for key, value in api_kwargs.items():
            if key in {"timeout", "extra_headers", "extra_query"}:
                continue
            if key == "extra_body":
                if not isinstance(value, Mapping):
                    return None
                request_body.update(value)
            else:
                request_body[key] = value
        serialized = json.dumps(
            request_body,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
    except Exception:
        # Audit instrumentation is deliberately fail-open.
        return None


def _log_retry_event_fail_open(
    *,
    request_id: Optional[str],
    session_id: Optional[str],
    endpoint: Optional[str],
    request_kind: str,
    attempt: int,
    max_attempts: int,
    exc: BaseException,
    failure_classification: str,
    model: str,
    messages: List[Dict[str, Any]],
    api_kwargs: Mapping[str, Any],
) -> None:
    """Emit one safe structured retry record; never affect generation."""
    try:
        payload_sha256 = _retry_payload_sha256(model, messages, api_kwargs)
        if request_id is None or payload_sha256 is None:
            return
        event = {
            "attempt": attempt,
            "endpoint": endpoint,
            "exception_class": type(exc).__name__,
            "failure_classification": failure_classification,
            "max_attempts": max_attempts,
            "payload_sha256": payload_sha256,
            "request_id": request_id,
            "request_kind": request_kind,
            "session_id": session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z"),
        }
        logger.warning(
            "[LLMClient-EVENT] %s",
            json.dumps(event, separators=(",", ":"), sort_keys=True),
        )
    except Exception:
        # Logging, clock, or serialization failures cannot change request flow.
        pass


def _managed_vllm_record_is_attested(
    instance: Dict[str, Any],
    *,
    model: str,
    expected_server_spec_digest: str,
    expected_topology_digest: str,
) -> bool:
    """Validate a managed registry record before generation selection.

    The immutable digests bind the record to the service manifest.  The health
    fields additionally verify that the process which published the record
    actually exposed the expected OpenAI-compatible model identity and that its
    realized launcher fields agree with the registry envelope.
    """
    if (
        not expected_server_spec_digest
        or not expected_topology_digest
        or instance.get("server_spec_digest") != expected_server_spec_digest
        or instance.get("topology_digest") != expected_topology_digest
        or instance.get("backend") != "vllm"
        or instance.get("protocol") != "openai-v1"
        or instance.get("effective_model") != model
    ):
        return False

    health = instance.get("health")
    served_models = health.get("served_models") if isinstance(health, dict) else None
    if not isinstance(health, dict) or (
        health.get("status") != "ok"
        or health.get("backend") != "vllm"
        or health.get("protocol") != "openai-v1"
        or health.get("effective_model") != model
        or not isinstance(served_models, list)
        or model not in served_models
    ):
        return False

    requested_model = instance.get("requested_model")
    if not requested_model or health.get("requested_model") != requested_model:
        return False
    requested_revision = instance.get("requested_model_revision")
    health_revision = health.get("requested_model_revision")
    managed_hf_model = (
        isinstance(requested_model, str)
        and "/" in requested_model
        and not os.path.isabs(requested_model)
        and not requested_model.startswith(".")
    )
    if managed_hf_model and (
        not isinstance(requested_revision, str)
        or _FROZEN_HF_REVISION_RE.fullmatch(requested_revision) is None
        or health_revision != requested_revision
    ):
        return False
    if not managed_hf_model and (
        requested_revision is not None or health_revision is not None
    ) and health_revision != requested_revision:
        return False

    for registry_key, health_key in (
        ("max_model_len", "max_model_len"),
        ("max_num_seqs", "max_num_seqs"),
        ("tp", "tensor_parallel_size"),
    ):
        try:
            if int(instance.get(registry_key)) != int(health.get(health_key)):
                return False
        except (TypeError, ValueError):
            return False

    for key in ("kv_cache_dtype", "quantization"):
        if key not in instance or key not in health or instance[key] != health[key]:
            return False

    registry_gpus = [str(value) for value in (instance.get("gpus") or [])]
    health_gpu = health.get("gpu")
    if not isinstance(health_gpu, dict) or not registry_gpus or registry_gpus != [
        str(value) for value in (health_gpu.get("selected_devices") or [])
    ]:
        return False
    return True


class LLMClient:
    """OpenAI-compatible LLM client with vLLM auto-discovery.

    When ``config.llm_base_url == 'vllm'``, endpoints are read from
    ``logs/serve.json`` and load-balanced with session-sticky routing.

    Features:
    - Client pooling: reuses AsyncOpenAI instances per endpoint
    - Periodic re-discovery: re-reads serve.json every 60s
    - Health tracking: marks failed endpoints as unhealthy for 60s
    - Sticky routing: hash(session_id) maps to consistent endpoint
    """

    _RETRYABLE_ERRORS = (
        asyncio.TimeoutError,
        ConnectionError,
        OSError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        NotFoundError,
    )

    # Subset of errors that indicate the server is completely unreachable
    _CONNECTION_ERRORS = (
        ConnectionError,
        OSError,
        APIConnectionError,
        APITimeoutError,
        NotFoundError,
    )

    def __init__(self, config: SpatialAgentConfig):
        self.config = config
        self._model = config.llm_model
        resolved_timeouts = resolve_request_timeouts(config)
        self._llm_request_timeout_sec = resolved_timeouts[
            "llm_request_timeout_sec"
        ]
        self._vlm_request_timeout_sec = resolved_timeouts[
            "vlm_request_timeout_sec"
        ]

        # Historical, unpinned runs keep forwarding an explicit environment
        # override as the SDK timeout. Pinned policies instead use a phase-aware
        # SDK timeout whose read phase is later than the explicit outer budget.
        if resolved_timeouts["llm_request_timeout_source"] == "config":
            self._llm_sdk_timeout = _sdk_transport_timeout_for_deadline(
                self._llm_request_timeout_sec
            )
        elif resolved_timeouts["llm_request_timeout_source"].startswith("env:"):
            self._llm_sdk_timeout = self._llm_request_timeout_sec
        else:
            self._llm_sdk_timeout = None

        if resolved_timeouts["vlm_request_timeout_source"] == "config":
            self._vlm_sdk_timeout = _sdk_transport_timeout_for_deadline(
                self._vlm_request_timeout_sec
            )
        elif resolved_timeouts["vlm_request_timeout_source"].startswith("env:"):
            self._vlm_sdk_timeout = self._vlm_request_timeout_sec
        else:
            self._vlm_sdk_timeout = None
        self._endpoints: List[str] = []
        self._api_key = (
            config.llm_api_key
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "bearer"
        )
        self._is_vllm = config.llm_base_url.lower().strip() == "vllm"
        self._lifecycle_discovery = _vllm_lifecycle_requested()
        # Managed generations publish a successor before retiring the old
        # endpoint.  A short registry TTL lets compliant clients leave the old
        # generation well inside the overlap without changing request payloads.
        self._discovery_ttl_sec = (
            1.0 if self._lifecycle_discovery else _DISCOVERY_TTL_SEC
        )

        # Client pool: endpoint URL -> AsyncOpenAI
        self._client_pool: Dict[str, AsyncOpenAI] = {}
        # Endpoints removed from discovery are not closed while a coroutine is
        # still using their transport.  They retire after the local in-flight
        # counter reaches zero.
        self._retiring_endpoints: set[str] = set()

        # Health tracking: endpoint URL -> timestamp when marked unhealthy
        self._unhealthy: Dict[str, float] = {}

        # Active request tracking for least-connections routing
        self._active_requests: Dict[str, int] = defaultdict(int)

        # Re-discovery state
        self._last_discovery: float = 0.0
        self._discovery_lock: Optional[asyncio.Lock] = None  # lazy init

        # Per-session token usage accumulator. Keyed by usage_session_id.
        # Tracks total thinking tokens and peak prompt-token (context length)
        # so workflow.arun() can attach usage stats to the agent result log.
        self._usage_lock = threading.Lock()
        self._session_usage: Dict[str, Dict[str, int]] = {}

        if self._is_vllm:
            self._endpoints = self._discover_with_wait()
            self._last_discovery = time.monotonic()
            # Health check on startup
            self._health_check_endpoints()
            print(f"[LLMClient] Found {len(self._endpoints)} vLLM endpoint(s) for {self._model}")
        else:
            self._endpoints = [config.llm_base_url]

    # ------------------------------------------------------------------
    # Pickle support (needed for cloudpickle injection into Jupyter kernel)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # AsyncOpenAI clients contain _thread.RLock and httpx sessions
        # that cannot be pickled — drop them; they'll be recreated lazily.
        state["_client_pool"] = {}
        state["_discovery_lock"] = None
        state["_active_requests"] = defaultdict(int)
        state["_retiring_endpoints"] = set()
        # threading.Lock is not picklable — recreate on the other side.
        state["_usage_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._usage_lock is None:
            self._usage_lock = threading.Lock()
        if not hasattr(self, "_retiring_endpoints"):
            self._retiring_endpoints = set()

    # ------------------------------------------------------------------
    # vLLM discovery
    # ------------------------------------------------------------------

    def _discover_vllm_endpoints(self) -> List[str]:
        """Read logs/serve.json and return all endpoints for the model."""
        # Latch managed mode if even one envelope key appears after client
        # construction.  Once latched, losing any key remains a fail-closed
        # condition for the lifetime of this client.
        if _vllm_lifecycle_requested():
            self._lifecycle_discovery = True
        managed_envelope = None
        if self._lifecycle_discovery:
            managed_envelope = _managed_vllm_envelope()
            if managed_envelope is None:
                return []

        serve_file = os.path.join(
            os.path.dirname(__file__), "..", "logs", "serve.json"
        )
        serve_file = os.path.abspath(serve_file)
        if not os.path.exists(serve_file):
            return []
        try:
            with open(serve_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            if self._lifecycle_discovery:
                raise RuntimeError(
                    f"Managed vLLM registry is unreadable or corrupt: {serve_file}"
                ) from exc
            return []

        if not isinstance(data, dict):
            if self._lifecycle_discovery:
                raise RuntimeError(
                    f"Managed vLLM registry root is not an object: {serve_file}"
                )
            return []

        model_instances = data.get(self._model, {})
        if not isinstance(model_instances, dict):
            if self._lifecycle_discovery:
                raise RuntimeError(
                    "Managed vLLM registry model bucket is not an object: "
                    f"{self._model!r}"
                )
            return []
        raw_allowlist = os.getenv("SPATIAL_AGENT_VLLM_ENDPOINT_ALLOWLIST")
        endpoint_allowlist: Optional[set[str]] = None
        if raw_allowlist:
            try:
                decoded = json.loads(raw_allowlist)
                endpoint_allowlist = (
                    {str(value) for value in decoded}
                    if isinstance(decoded, list) else set()
                )
            except json.JSONDecodeError:
                logger.error(
                    "Malformed SPATIAL_AGENT_VLLM_ENDPOINT_ALLOWLIST; "
                    "failing discovery closed"
                )
                endpoint_allowlist = set()
        required_pool = (
            managed_envelope["SPATIAL_AGENT_VLLM_POOL_ID"]
            if managed_envelope is not None else ""
        )
        required_namespace = (
            managed_envelope["SPATIAL_AGENT_VLLM_NAMESPACE"]
            if managed_envelope is not None else ""
        )
        expected_server_spec_digest = (
            managed_envelope["SPATIAL_AGENT_VLLM_SERVER_SPEC_DIGEST"]
            if managed_envelope is not None else ""
        )
        expected_topology_digest = (
            managed_envelope["SPATIAL_AGENT_VLLM_TOPOLOGY_DIGEST"]
            if managed_envelope is not None else ""
        )
        raw_slots = os.getenv("SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST")
        slot_allowlist: Optional[set[str]] = None
        if managed_envelope is not None:
            slot_allowlist = managed_envelope["slots"]
        elif raw_slots:
            try:
                decoded = json.loads(raw_slots)
                slot_allowlist = (
                    {str(value) for value in decoded}
                    if isinstance(decoded, list) else set()
                )
            except json.JSONDecodeError:
                logger.error(
                    "Malformed SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST; "
                    "failing discovery closed"
                )
                slot_allowlist = set()
        candidates: List[Tuple[str, int, str, str]] = []
        for uid, instance in model_instances.items():
            if not isinstance(instance, dict):
                continue
            # A lifecycle-managed pool is owner-gated.  An unscoped legacy or
            # direct client must not silently consume it because that client
            # has no lease and teardown cannot account for its requests.
            if instance.get("pool_id") and not self._lifecycle_discovery:
                continue
            if required_pool and instance.get("pool_id") != required_pool:
                continue
            if required_namespace and instance.get("namespace") != required_namespace:
                continue
            if self._lifecycle_discovery:
                try:
                    attested = _managed_vllm_record_is_attested(
                        instance,
                        model=self._model,
                        expected_server_spec_digest=expected_server_spec_digest,
                        expected_topology_digest=expected_topology_digest,
                    )
                except (TypeError, ValueError):
                    # A malformed successor must not hide an otherwise valid
                    # generation for the same frozen slot.
                    attested = False
                if not attested:
                    continue
            if instance.get("state", "ready") not in {"ready", "accepting"}:
                continue
            if instance.get("draining") or instance.get("accepting") is False:
                continue
            slot_id = str(instance.get("slot_id") or uid)
            if slot_allowlist is not None and slot_id not in slot_allowlist:
                continue
            ip = instance.get("ip")
            port = instance.get("port")
            try:
                port_number = int(port)
                generation = int(instance.get("generation") or 0)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(ip, str)
                or not ip
                or isinstance(port, bool)
                or not 1 <= port_number <= 65535
            ):
                continue
            url_host = (
                f"[{ip}]"
                if ":" in ip and not ip.startswith("[")
                else ip
            )
            endpoint = f"http://{url_host}:{port_number}/v1"
            if endpoint_allowlist is not None and endpoint not in endpoint_allowlist:
                continue
            candidates.append(
                (slot_id, generation, str(uid), endpoint)
            )

        if self._lifecycle_discovery:
            candidates_by_slot: Dict[str, List[Tuple[int, str, str]]] = {}
            for slot_id, generation, uid, endpoint in candidates:
                candidates_by_slot.setdefault(slot_id, []).append(
                    (generation, uid, endpoint)
                )
            required_slots = (
                slot_allowlist
                if slot_allowlist is not None
                else set(candidates_by_slot)
            )
            if set(candidates_by_slot) != required_slots:
                return []

            # Resolve a generation only after accounting for client-observed
            # health.  Otherwise a dead newest endpoint would be selected and
            # then dropped, shadowing a still-healthy predecessor in the same
            # authorized pool/namespace/digest/slot.
            now = time.monotonic()
            unhealthy = getattr(self, "_unhealthy", {})
            selected: List[str] = []
            for slot_id in sorted(required_slots):
                endpoint = next(
                    (
                        candidate_endpoint
                        for _generation, _uid, candidate_endpoint in sorted(
                            candidates_by_slot[slot_id],
                            key=lambda item: (item[0], item[1]),
                            reverse=True,
                        )
                        if candidate_endpoint not in unhealthy
                        or now - unhealthy[candidate_endpoint]
                        >= _UNHEALTHY_COOLDOWN_SEC
                    ),
                    None,
                )
                if endpoint is None:
                    return []
                selected.append(endpoint)
            return selected

        return sorted(endpoint for _, _, _, endpoint in candidates)

    def _discover_with_wait(self) -> List[str]:
        """Discover vLLM endpoints, waiting up to 4 hours if none found."""
        deadline = time.monotonic() + _SERVER_WAIT_TIMEOUT_SEC
        waited = False
        while True:
            endpoints = self._discover_vllm_endpoints()
            if endpoints:
                if waited:
                    print(f"[LLMClient] Server discovered after waiting.")
                return endpoints

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"No vLLM instances found for model '{self._model}' "
                    f"in serve.json after waiting {_SERVER_WAIT_TIMEOUT_SEC // 3600}h. "
                    f"Is the vLLM service running?"
                )

            if not waited:
                print(
                    f"[LLMClient] No vLLM endpoints for '{self._model}' yet. "
                    f"Waiting up to {remaining / 3600:.1f}h for server to start..."
                )
                waited = True
            else:
                logger.info(
                    "[LLMClient] Still waiting for server... (%.0f min remaining)",
                    remaining / 60,
                )
            time.sleep(_SERVER_WAIT_POLL_SEC)

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def _managed_health_checked_generation(
        self,
        endpoints: List[str],
        *,
        trusted_endpoints: Optional[set[str]] = None,
    ) -> List[str]:
        """Return a complete reachable generation, falling back per slot.

        ``trusted_endpoints`` are endpoints already serving this client.  A
        handoff only probes newly discovered endpoints, so periodic discovery
        adds no traffic to the active generation.  Failed probes feed the same
        cooldown map as real connection failures, allowing registry discovery
        to resolve the next authorized generation for that logical slot.
        """
        trusted = trusted_endpoints or set()
        attempted_generations: set[Tuple[str, ...]] = set()
        candidate_generation = list(endpoints)
        while candidate_generation:
            generation = tuple(candidate_generation)
            if generation in attempted_generations:
                break
            attempted_generations.add(generation)
            failed = []
            for ep in candidate_generation:
                if ep in trusted:
                    continue
                url = f"{ep}/models"
                try:
                    req = urllib.request.Request(url, method="GET")
                    response = urllib.request.urlopen(
                        req, timeout=_HEALTH_CHECK_TIMEOUT_SEC
                    )
                    payload = json.loads(response.read().decode("utf-8"))
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if (
                        not isinstance(data, list)
                        or not any(
                            isinstance(entry, dict)
                            and entry.get("id") == self._model
                            for entry in data
                        )
                    ):
                        raise RuntimeError(
                            f"/v1/models does not serve exact model {self._model!r}"
                        )
                except Exception as exc:
                    logger.warning(
                        "[LLMClient] Health check failed for %s: %s",
                        ep,
                        exc,
                    )
                    failed.append(ep)
            if not failed:
                return candidate_generation
            for ep in failed:
                self._mark_unhealthy(ep)
            replacement = self._discover_vllm_endpoints()
            if not replacement or tuple(replacement) in attempted_generations:
                break
            logger.info(
                "[LLMClient] Falling back to prior healthy managed "
                "generation for %d failed endpoint(s)",
                len(failed),
            )
            candidate_generation = replacement
        return []

    def _health_check_endpoints(self) -> None:
        """Probe each endpoint at startup, drop unreachable ones."""
        if not self._is_vllm or not self._endpoints:
            return

        if self._lifecycle_discovery:
            self._endpoints = self._managed_health_checked_generation(
                self._endpoints
            )
            if not self._endpoints:
                logger.error(
                    "[LLMClient] Managed topology has no complete healthy "
                    "generation; failing closed"
                )
            return

        healthy = []
        for ep in self._endpoints:
            url = f"{ep}/models"
            try:
                req = urllib.request.Request(url, method="GET")
                urllib.request.urlopen(req, timeout=_HEALTH_CHECK_TIMEOUT_SEC)
                healthy.append(ep)
            except Exception as exc:
                logger.warning(
                    "[LLMClient] Health check failed for %s: %s", ep, exc,
                )

        if healthy:
            dropped = len(self._endpoints) - len(healthy)
            if dropped:
                logger.info(
                    "[LLMClient] Health check: %d/%d endpoints reachable (dropped %d)",
                    len(healthy), len(self._endpoints), dropped,
                )
            self._endpoints = healthy
        else:
            logger.warning(
                "[LLMClient] All %d endpoints failed health check — keeping all",
                len(self._endpoints),
            )

    # ------------------------------------------------------------------
    # Periodic re-discovery
    # ------------------------------------------------------------------

    async def _maybe_rediscover(self) -> None:
        """Re-read serve.json if TTL has expired. Thread-safe via asyncio.Lock."""
        if not self._is_vllm:
            return

        now = time.monotonic()
        if now - self._last_discovery < self._discovery_ttl_sec:
            return

        # Lazy init lock (must be created in async context)
        if self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()

        async with self._discovery_lock:
            # Double-check after acquiring lock
            if time.monotonic() - self._last_discovery < self._discovery_ttl_sec:
                return

            new_endpoints = self._discover_vllm_endpoints()
            if not new_endpoints and not self._lifecycle_discovery:
                self._last_discovery = time.monotonic()
                return

            if (
                self._lifecycle_discovery
                and new_endpoints != self._endpoints
                and new_endpoints
            ):
                new_endpoints = self._managed_health_checked_generation(
                    new_endpoints,
                    trusted_endpoints=set(self._endpoints),
                )

            if new_endpoints != self._endpoints:
                # Retire removed endpoints.  Never close a transport while a
                # request coroutine is still using it.
                removed = set(self._endpoints) - set(new_endpoints)
                for ep in removed:
                    self._unhealthy.pop(ep, None)
                    if self._active_requests.get(ep, 0) > 0:
                        self._retiring_endpoints.add(ep)
                    else:
                        client = self._client_pool.pop(ep, None)
                        if client:
                            try:
                                await client.close()
                            except Exception:
                                pass
                        self._active_requests.pop(ep, None)

                added = set(new_endpoints) - set(self._endpoints)
                if added or removed:
                    logger.info(
                        "[LLMClient] Endpoints updated: +%d -%d (total %d)",
                        len(added), len(removed), len(new_endpoints),
                    )

                self._endpoints = new_endpoints

            # Clear expired unhealthy entries
            cutoff = time.monotonic() - _UNHEALTHY_COOLDOWN_SEC
            if self._lifecycle_discovery:
                # Retain failed successor endpoints through their cooldown even
                # while a predecessor is selected; otherwise every one-second
                # registry refresh would probe and reconsider the same dead
                # generation.
                self._unhealthy = {
                    ep: ts for ep, ts in self._unhealthy.items()
                    if ts > cutoff
                }
            else:
                self._unhealthy = {
                    ep: ts for ep, ts in self._unhealthy.items()
                    if ts > cutoff and ep in self._endpoints
                }

            self._last_discovery = time.monotonic()

    # ------------------------------------------------------------------
    # Endpoint selection
    # ------------------------------------------------------------------

    def _get_healthy_endpoints(self) -> List[str]:
        """Return endpoints not marked unhealthy. Falls back to all if none healthy."""
        now = time.monotonic()
        healthy = [
            ep for ep in self._endpoints
            if ep not in self._unhealthy
            or now - self._unhealthy[ep] >= _UNHEALTHY_COOLDOWN_SEC
        ]
        if self._lifecycle_discovery:
            return healthy
        return healthy if healthy else self._endpoints

    def _pick_endpoint(self, session_id: Optional[str] = None) -> str:
        """Pick endpoint using least-connections, with optional sticky preference."""
        candidates = self._get_healthy_endpoints()
        if self._lifecycle_discovery and len(candidates) != len(self._endpoints):
            # Do not silently route a partial frozen topology. Re-resolve each
            # slot so an unhealthy successor can fall back to its predecessor.
            candidates = self._discover_vllm_endpoints()
        if not candidates and not self._lifecycle_discovery:
            candidates = self._endpoints

        if not candidates:
            raise ConnectionError(
                f"No accepting managed vLLM endpoint for model {self._model!r}"
            )

        if len(candidates) == 1:
            return candidates[0]

        if session_id and self._is_vllm:
            # Prefer sticky endpoint for prefix cache, unless overloaded
            preferred_idx = hash(session_id) % len(candidates)
            preferred = candidates[preferred_idx]
            min_load = min(self._active_requests[ep] for ep in candidates)
            if self._active_requests[preferred] <= max(min_load * 2, min_load + 2):
                return preferred

        # Least-connections with random tie-breaking
        min_load = min(self._active_requests[ep] for ep in candidates)
        least_loaded = [ep for ep in candidates if self._active_requests[ep] == min_load]
        return random.choice(least_loaded)

    def _mark_unhealthy(self, endpoint: str) -> None:
        """Mark an endpoint as unhealthy after a connection failure."""
        self._unhealthy[endpoint] = time.monotonic()
        logger.info("[LLMClient] Marked endpoint unhealthy: %s", endpoint)

    def _log_transport_failure_diagnostic(
        self,
        *,
        request_kind: str,
        exc: BaseException,
        endpoint: Optional[str],
        attempt: int,
        max_retries: int,
        elapsed_sec: float,
        sdk_timeout: Any,
    ) -> None:
        """Log safe phase evidence for the model-opt-in transport path."""
        if not self.config.offload_multimodal_encoding:
            return
        logger.warning(
            "[LLMClient] %s transport diagnostic: endpoint=%s "
            "attempt=%d/%d elapsed_sec=%.3f exception_chain=%s sdk_timeout=%r",
            request_kind,
            endpoint or "<unselected>",
            attempt,
            max_retries,
            elapsed_sec,
            _exception_type_chain(exc),
            sdk_timeout,
        )

    def _track_request_start(self, endpoint: str) -> None:
        """Increment active request counter for an endpoint."""
        self._active_requests[endpoint] += 1

    async def _track_request_end(self, endpoint: str) -> None:
        """Decrement active count and finish a deferred transport close."""
        self._active_requests[endpoint] = max(0, self._active_requests[endpoint] - 1)
        if (
            self._active_requests[endpoint] == 0
            and endpoint in self._retiring_endpoints
        ):
            self._retiring_endpoints.discard(endpoint)
            client = self._client_pool.pop(endpoint, None)
            if client:
                try:
                    await client.close()
                except Exception:
                    pass
            self._active_requests.pop(endpoint, None)

    # ------------------------------------------------------------------
    # Client pool
    # ------------------------------------------------------------------

    def _get_client(self, endpoint: str) -> AsyncOpenAI:
        """Get or create a pooled AsyncOpenAI client for the endpoint."""
        client = self._client_pool.get(endpoint)
        if client is None:
            client_kwargs: Dict[str, Any] = dict(
                api_key=self._api_key,
                base_url=endpoint,
            )
            sdk_max_retries = getattr(self.config, "llm_sdk_max_retries", None)
            if sdk_max_retries is not None:
                client_kwargs["max_retries"] = sdk_max_retries
            client = AsyncOpenAI(**client_kwargs)
            self._client_pool[endpoint] = client
        return client

    @staticmethod
    def _lifecycle_headers() -> Dict[str, str]:
        """Out-of-band attribution headers; never part of model messages."""
        result: Dict[str, str] = {}
        for header, env_name in (
            ("X-Spatial-Agent-Owner", "SPATIAL_AGENT_OWNER_ID"),
            ("X-Spatial-Agent-Cohort", "SPATIAL_AGENT_COHORT_ID"),
            ("X-Spatial-Agent-Lease", "SPATIAL_AGENT_LEASE_ID"),
            (
                "X-Spatial-Agent-Manifest",
                "SPATIAL_AGENT_SERVICE_MANIFEST_DIGEST",
            ),
        ):
            value = os.getenv(env_name)
            if value:
                result[header] = value
        return result

    @classmethod
    def _merge_lifecycle_headers(
        cls,
        existing: Optional[Mapping[str, Any]],
        lifecycle: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Merge attribution headers without overriding caller intent."""
        if existing is None:
            merged: Dict[str, Any] = {}
        elif isinstance(existing, Mapping):
            merged = dict(existing)
        else:
            raise TypeError("extra_headers must be a mapping")

        existing_by_lower = {str(name).lower(): name for name in merged}
        for header, value in lifecycle.items():
            normalized = header.lower()
            if normalized not in _RESERVED_LIFECYCLE_HEADERS:
                raise ValueError(f"Unexpected lifecycle header {header!r}")
            caller_name = existing_by_lower.get(normalized)
            if caller_name is not None:
                if str(merged[caller_name]) != value:
                    raise ValueError(
                        f"Conflicting reserved lifecycle header {header!r}"
                    )
                continue
            merged[header] = value
        return merged

    def _apply_lifecycle_headers(
        self, api_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        lifecycle = self._lifecycle_headers()
        if not lifecycle:
            return api_kwargs
        result = dict(api_kwargs)
        result["extra_headers"] = self._merge_lifecycle_headers(
            api_kwargs.get("extra_headers"), lifecycle,
        )
        return result

    async def close(self) -> None:
        """Close all pooled clients. Call on shutdown."""
        for ep, client in self._client_pool.items():
            try:
                await client.close()
            except Exception:
                pass
        self._client_pool.clear()
        self._retiring_endpoints.clear()

    # ------------------------------------------------------------------
    # Token usage tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response) -> Dict[str, int]:
        """Pull token counts from a chat-completions response.

        Returns a dict with prompt_tokens, completion_tokens, reasoning_tokens
        (best effort; vLLM exposes reasoning tokens via
        ``completion_tokens_details.reasoning_tokens`` on newer builds).
        Missing fields default to 0.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def _record_usage(self, usage_session_id: Optional[str], response) -> None:
        """Accumulate token counts for *usage_session_id*. No-op when None."""
        if not usage_session_id:
            return
        u = self._extract_usage(response)
        with self._usage_lock:
            stats = self._session_usage.get(usage_session_id)
            if stats is None:
                stats = {
                    "num_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_reasoning_tokens": 0,
                    "max_prompt_tokens": 0,
                    "max_completion_tokens": 0,
                }
                self._session_usage[usage_session_id] = stats
            stats["num_calls"] += 1
            stats["total_prompt_tokens"] += u["prompt_tokens"]
            stats["total_completion_tokens"] += u["completion_tokens"]
            stats["total_reasoning_tokens"] += u["reasoning_tokens"]
            if u["prompt_tokens"] > stats["max_prompt_tokens"]:
                stats["max_prompt_tokens"] = u["prompt_tokens"]
            if u["completion_tokens"] > stats["max_completion_tokens"]:
                stats["max_completion_tokens"] = u["completion_tokens"]

    def pop_session_usage(self, usage_session_id: str) -> Dict[str, int]:
        """Remove and return accumulated usage stats for a session.

        Returns a zero-filled dict when the session was never recorded
        (e.g. no usage_session_id was passed to generate()).
        """
        with self._usage_lock:
            stats = self._session_usage.pop(usage_session_id, None)
        if stats is None:
            return {
                "num_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_reasoning_tokens": 0,
                "max_prompt_tokens": 0,
                "max_completion_tokens": 0,
            }
        return stats

    # ------------------------------------------------------------------
    # API kwargs builder
    # ------------------------------------------------------------------

    def _build_api_kwargs(self, params: LLMRoleParams) -> Dict[str, Any]:
        """Build kwargs for chat.completions.create from role params.

        Always includes: model, max_tokens, temperature.
        Standard OpenAI params if set: top_p, presence_penalty, seed.
        vLLM-only params in extra_body if set: top_k, min_p,
        repetition_penalty, enable_thinking.
        """
        kwargs: Dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        # Standard OpenAI-compatible params
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.presence_penalty is not None:
            kwargs["presence_penalty"] = params.presence_penalty
        if params.seed is not None:
            kwargs["seed"] = params.seed

        # vLLM-specific params via extra_body
        if self._is_vllm:
            extra_body: Dict[str, Any] = {}
            if params.top_k is not None:
                extra_body["top_k"] = params.top_k
            if params.min_p is not None:
                extra_body["min_p"] = params.min_p
            if params.repetition_penalty is not None:
                extra_body["repetition_penalty"] = params.repetition_penalty
            if params.enable_thinking is not None:
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": params.enable_thinking
                }
            if extra_body:
                kwargs["extra_body"] = extra_body

        return kwargs

    @staticmethod
    def _normalize_multimodal_order(
        model: str,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Put media before text in Cosmos3 user messages.

        Cosmos3's prompt format expects visual inputs before user text. The
        stable sort preserves media order and text order, and copies only the
        affected message/content containers so LangGraph state is unchanged.
        """
        if "cosmos3" not in model.lower():
            return messages

        media_types = {"image_url", "video_url", "image", "video"}
        normalized = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, list):
                content = sorted(
                    content,
                    key=lambda part: not (
                        isinstance(part, dict)
                        and part.get("type") in media_types
                    ),
                )
                message = {**message, "content": content}
            normalized.append(message)
        return normalized

    # ------------------------------------------------------------------
    # Gemma-4 thinking workaround
    # ------------------------------------------------------------------

    @staticmethod
    def _fixup_gemma_thinking(
        content: str, reasoning: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Split Gemma-4 thinking from content when the reasoning parser fails.

        vLLM's Gemma4 reasoning parser relies on ``<|channel>`` /
        ``<channel|>`` special tokens in the decoded text, but
        ``skip_special_tokens=True`` (the default) strips them before the
        non-streaming ``extract_reasoning`` path sees them.  The result is
        ``reasoning=None`` with the full output (``thought\\n<thinking>\\n
        <content>``) in *content*.

        This method detects that case and splits thinking from content by
        finding the ``thought\\n`` prefix produced by Gemma-4's channel
        role label.
        """
        if reasoning is not None or not content.startswith("thought\n"):
            return content, reasoning

        # The thinking is everything after "thought\n" up to the actual
        # response.  Since the channel delimiter is gone, we heuristically
        # split at the last double-newline before recognizable content
        # structure (bold headers, markdown headings).  This isn't perfect
        # but beats leaking the full thinking blob.
        import re
        # Try in order: **Purpose** (agent step), ## / ### header, **bold
        # heading at line start.
        m = (re.search(r"\*\*Purpose\*\*", content)
             or re.search(r"\n(#{2,3}\s)", content)
             or re.search(r"\n(\*\*\w)", content))
        if m:
            reasoning = content[len("thought\n"):m.start()].strip()
            content = content[m.start():].strip()
        else:
            # Can't find boundary — treat entire body as content,
            # just strip the "thought\n" prefix.
            reasoning = None
            content = content[len("thought\n"):].strip()

        return content, reasoning

    # ------------------------------------------------------------------
    # Main agent generation (TEXT-ONLY, no images)
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        role_params: Optional[LLMRoleParams] = None,
        max_retries: int = 5,
        session_id: Optional[str] = None,
        usage_session_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """LLM call for the main agent.

        ``messages`` may contain multimodal content (image_url parts) when
        key frames are injected in the first user message.
        Returns ``(content, reasoning_content)`` where *reasoning_content*
        may be ``None`` for non-thinking models.

        Uses sticky routing when ``session_id`` is provided — the same
        session consistently hits the same vLLM server for prefix cache hits.

        Retries up to *max_retries* times on transient errors with exponential
        backoff.  If all retries fail due to connection errors (server down),
        waits up to 4 hours for the server to come back before giving up.
        """
        params = role_params or self.config.main_params
        api_kwargs = self._build_api_kwargs(params)
        api_kwargs = self._apply_lifecycle_headers(api_kwargs)
        if self._llm_sdk_timeout is not None:
            api_kwargs["timeout"] = self._llm_sdk_timeout
        messages = self._normalize_multimodal_order(self._model, messages)
        retry_request_id = _retry_request_id_fail_open()

        # Proactive re-discovery before picking endpoint
        await self._maybe_rediscover()

        wait_start: Optional[float] = None
        timeout_failures = 0

        while True:
            last_exc: Optional[Exception] = None
            is_connection_failure = False
            last_was_generation_timeout = False

            for attempt in range(max_retries):
                attempt_started = time.monotonic()
                endpoint: Optional[str] = None
                try:
                    endpoint = self._pick_endpoint(session_id)
                    client = self._get_client(endpoint)
                    self._track_request_start(endpoint)
                    request_kwargs = {
                        "model": self._model,
                        "messages": messages,
                        **api_kwargs,
                    }
                    response = await asyncio.wait_for(
                        _with_transport_timeout_identity(
                            client.chat.completions.create(**request_kwargs)
                        ),
                        timeout=self._llm_request_timeout_sec,
                    )
                except self._RETRYABLE_ERRORS as exc:
                    last_exc = exc
                    timeout_retry_limit = getattr(
                        self.config, "llm_timeout_retries", None
                    )
                    is_scoped_timeout = (
                        timeout_retry_limit is not None
                        and isinstance(exc, asyncio.TimeoutError)
                        and not isinstance(exc, _TransportAsyncTimeoutError)
                    )
                    if is_scoped_timeout:
                        # Only our explicit outer wait_for deadline reaches this
                        # branch. SDK/transport timeouts stay retryable below.
                        timeout_failures += 1
                        is_connection_failure = False
                        last_was_generation_timeout = True
                        if timeout_failures > timeout_retry_limit:
                            logger.warning(
                                "[LLMClient] Generation timed out after %d attempt(s); "
                                "configured timeout retries exhausted.",
                                timeout_failures,
                            )
                            raise LLMGenerationTimeoutError(
                                "LLM", timeout_failures
                            ) from exc
                    else:
                        is_connection_failure = isinstance(
                            exc, self._CONNECTION_ERRORS
                        )
                        last_was_generation_timeout = False
                    if is_connection_failure and endpoint is not None:
                        self._mark_unhealthy(endpoint)
                    logger.warning(
                        "[LLMClient] Attempt %d/%d failed (%s: %s), retrying...",
                        attempt + 1, max_retries, type(exc).__name__, exc,
                    )
                    _log_retry_event_fail_open(
                        request_id=retry_request_id,
                        session_id=(
                            usage_session_id
                            if usage_session_id is not None else session_id
                        ),
                        endpoint=endpoint,
                        request_kind="LLM",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        exc=exc,
                        failure_classification=(
                            "generation" if is_scoped_timeout else
                            "transport" if is_connection_failure else "other"
                        ),
                        model=self._model,
                        messages=messages,
                        api_kwargs=api_kwargs,
                    )
                    if is_connection_failure:
                        self._log_transport_failure_diagnostic(
                            request_kind="LLM",
                            exc=exc,
                            endpoint=endpoint,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            elapsed_sec=time.monotonic() - attempt_started,
                            sdk_timeout=self._llm_sdk_timeout,
                        )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                except Exception as exc:
                    last_exc = exc
                    is_connection_failure = False
                    if attempt < max_retries - 1:
                        logger.warning(
                            "[LLMClient] Attempt %d/%d failed (%s: %s), retrying...",
                            attempt + 1, max_retries, type(exc).__name__, exc,
                        )
                        _log_retry_event_fail_open(
                            request_id=retry_request_id,
                            session_id=(
                                usage_session_id
                                if usage_session_id is not None else session_id
                            ),
                            endpoint=endpoint,
                            request_kind="LLM",
                            attempt=attempt + 1,
                            max_attempts=max_retries,
                            exc=exc,
                            failure_classification="other",
                            model=self._model,
                            messages=messages,
                            api_kwargs=api_kwargs,
                        )
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    raise
                finally:
                    if endpoint is not None:
                        await self._track_request_end(endpoint)

                # Guard against vLLM returning empty choices (transient server issue)
                if not response.choices:
                    last_exc = RuntimeError("vLLM returned empty choices")
                    is_connection_failure = True
                    logger.warning(
                        "[LLMClient] Attempt %d/%d: empty choices, retrying...",
                        attempt + 1, max_retries,
                    )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue

                # Success — extract content
                wait_start = None  # reset on success
                self._record_usage(usage_session_id, response)
                choice = response.choices[0]
                content = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None)

                if not content.strip() and reasoning:
                    logger.info(
                        "[LLMClient] content is empty, falling back to reasoning_content"
                    )
                    content = reasoning

                content, reasoning = self._fixup_gemma_thinking(
                    content, reasoning,
                )
                return content, reasoning

            # All quick retries exhausted
            if last_was_generation_timeout:
                timeout_retry_limit = getattr(
                    self.config, "llm_timeout_retries", None
                )
                if (
                    timeout_retry_limit is not None
                    and timeout_failures <= timeout_retry_limit
                ):
                    # Non-timeout failures may have consumed slots in the
                    # quick-retry loop. Start another bounded round so the
                    # scoped budget counts explicit outer deadlines, not
                    # unrelated transport/server failures.
                    continue
                raise LLMGenerationTimeoutError("LLM", timeout_failures) from last_exc
            if not is_connection_failure:
                raise last_exc  # type: ignore[misc]

            # Enforce server-wait timeout
            if wait_start is None:
                wait_start = time.monotonic()
            elif time.monotonic() - wait_start > _SERVER_WAIT_TIMEOUT_SEC:
                logger.error(
                    "[LLMClient] Server wait timeout (%.0fh). Giving up.",
                    _SERVER_WAIT_TIMEOUT_SEC / 3600,
                )
                raise last_exc  # type: ignore[misc]

            # Server appears down — wait and retry
            # vLLM: re-discovers endpoints from serve.json
            # External APIs: sleeps and retries the same endpoint
            logger.warning(
                "[LLMClient] Server unreachable after %d retries. "
                "Waiting %ds before retrying...", max_retries, _SERVER_WAIT_POLL_SEC,
            )
            await self._maybe_rediscover()
            await asyncio.sleep(_SERVER_WAIT_POLL_SEC)

    # ------------------------------------------------------------------
    # Isolated VLM session (for vlm.locate / vlm.ask_with_thinking)
    # ------------------------------------------------------------------

    async def generate_vision_query(
        self,
        images: List[Image.Image],
        question: str,
        system_prompt: str,
        role_params: Optional[LLMRoleParams] = None,
        session_id: Optional[str] = None,
        usage_session_id: Optional[str] = None,
    ) -> str:
        """Isolated VLM call: images + question -> text answer.

        This is completely separate from the main agent conversation.
        Uses sticky routing when ``session_id`` is provided.
        Retries on transient errors and waits for server if unavailable.
        """
        params = role_params or self.config.vlm_params
        api_kwargs = self._build_api_kwargs(params)
        api_kwargs = self._apply_lifecycle_headers(api_kwargs)
        if self._vlm_sdk_timeout is not None:
            api_kwargs["timeout"] = self._vlm_sdk_timeout

        # Build the exact historical multimodal payload.  Model profiles may
        # move the CPU-heavy batch encoding off the event loop so concurrent
        # requests cannot starve already-started HTTP connections.  One batch
        # is offloaded as a unit to preserve image order deterministically.
        if self.config.offload_multimodal_encoding:
            image_urls = await asyncio.to_thread(_images_to_base64_urls, images)
        else:
            image_urls = _images_to_base64_urls(images)

        user_content: List[Dict[str, Any]] = []
        for image_url in image_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })
        user_content.append({"type": "text", "text": question})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        retry_request_id = _retry_request_id_fail_open()

        # Proactive re-discovery before picking endpoint
        await self._maybe_rediscover()

        max_retries = 5
        wait_start: Optional[float] = None
        timeout_failures = 0

        while True:
            last_exc: Optional[Exception] = None
            is_connection_failure = False
            last_was_generation_timeout = False

            for attempt in range(max_retries):
                attempt_started = time.monotonic()
                endpoint: Optional[str] = None
                try:
                    endpoint = self._pick_endpoint(session_id)
                    client = self._get_client(endpoint)
                    self._track_request_start(endpoint)
                    request_kwargs = {
                        "model": self._model,
                        "messages": messages,
                        **api_kwargs,
                    }
                    response = await asyncio.wait_for(
                        _with_transport_timeout_identity(
                            client.chat.completions.create(**request_kwargs)
                        ),
                        timeout=self._vlm_request_timeout_sec,
                    )
                except self._RETRYABLE_ERRORS as exc:
                    last_exc = exc
                    timeout_retry_limit = getattr(
                        self.config, "llm_timeout_retries", None
                    )
                    is_scoped_timeout = (
                        timeout_retry_limit is not None
                        and isinstance(exc, asyncio.TimeoutError)
                        and not isinstance(exc, _TransportAsyncTimeoutError)
                    )
                    if is_scoped_timeout:
                        timeout_failures += 1
                        is_connection_failure = False
                        last_was_generation_timeout = True
                        if timeout_failures > timeout_retry_limit:
                            logger.warning(
                                "[LLMClient] VLM generation timed out after %d "
                                "attempt(s); configured timeout retries exhausted.",
                                timeout_failures,
                            )
                            raise LLMGenerationTimeoutError(
                                "VLM", timeout_failures
                            ) from exc
                    else:
                        is_connection_failure = isinstance(
                            exc, self._CONNECTION_ERRORS
                        )
                        last_was_generation_timeout = False
                    if is_connection_failure and endpoint is not None:
                        self._mark_unhealthy(endpoint)
                    logger.warning(
                        "[LLMClient] VLM attempt %d/%d failed (%s: %s), retrying...",
                        attempt + 1, max_retries, type(exc).__name__, exc,
                    )
                    _log_retry_event_fail_open(
                        request_id=retry_request_id,
                        session_id=(
                            usage_session_id
                            if usage_session_id is not None else session_id
                        ),
                        endpoint=endpoint,
                        request_kind="VLM",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        exc=exc,
                        failure_classification=(
                            "generation" if is_scoped_timeout else
                            "transport" if is_connection_failure else "other"
                        ),
                        model=self._model,
                        messages=messages,
                        api_kwargs=api_kwargs,
                    )
                    if is_connection_failure:
                        self._log_transport_failure_diagnostic(
                            request_kind="VLM",
                            exc=exc,
                            endpoint=endpoint,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            elapsed_sec=time.monotonic() - attempt_started,
                            sdk_timeout=self._vlm_sdk_timeout,
                        )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                except Exception as exc:
                    last_exc = exc
                    is_connection_failure = False
                    if attempt < max_retries - 1:
                        logger.warning(
                            "[LLMClient] VLM attempt %d/%d failed (%s: %s), retrying...",
                            attempt + 1, max_retries, type(exc).__name__, exc,
                        )
                        _log_retry_event_fail_open(
                            request_id=retry_request_id,
                            session_id=(
                                usage_session_id
                                if usage_session_id is not None else session_id
                            ),
                            endpoint=endpoint,
                            request_kind="VLM",
                            attempt=attempt + 1,
                            max_attempts=max_retries,
                            exc=exc,
                            failure_classification="other",
                            model=self._model,
                            messages=messages,
                            api_kwargs=api_kwargs,
                        )
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    raise
                finally:
                    if endpoint is not None:
                        await self._track_request_end(endpoint)

                # Guard against vLLM returning empty choices (transient server issue)
                if not response.choices:
                    last_exc = RuntimeError("vLLM returned empty choices")
                    is_connection_failure = True
                    logger.warning(
                        "[LLMClient] VLM attempt %d/%d: empty choices, retrying...",
                        attempt + 1, max_retries,
                    )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue

                wait_start = None  # reset on success
                self._record_usage(usage_session_id, response)
                choice = response.choices[0]
                content = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None)

                if not content.strip() and reasoning:
                    content = reasoning

                # Fix Gemma-4 thinking leak (skip_special_tokens strips
                # channel delimiters before the reasoning parser sees them).
                content, _ = self._fixup_gemma_thinking(content, reasoning)

                # Strip <think>...</think> tags that some vLLM versions
                # embed directly in content instead of reasoning_content.
                content = LLMResponseValidator._strip_thinking(content)

                return content

            # All quick retries exhausted
            if last_was_generation_timeout:
                timeout_retry_limit = getattr(
                    self.config, "llm_timeout_retries", None
                )
                if (
                    timeout_retry_limit is not None
                    and timeout_failures <= timeout_retry_limit
                ):
                    continue
                raise LLMGenerationTimeoutError("VLM", timeout_failures) from last_exc
            if not is_connection_failure:
                raise last_exc  # type: ignore[misc]

            # Enforce server-wait timeout
            if wait_start is None:
                wait_start = time.monotonic()
            elif time.monotonic() - wait_start > _SERVER_WAIT_TIMEOUT_SEC:
                logger.error(
                    "[LLMClient] VLM server wait timeout (%.0fh). Giving up.",
                    _SERVER_WAIT_TIMEOUT_SEC / 3600,
                )
                raise last_exc  # type: ignore[misc]

            # Server appears down — wait and retry
            logger.warning(
                "[LLMClient] VLM server unreachable after %d retries. "
                "Waiting %ds before retrying...", max_retries, _SERVER_WAIT_POLL_SEC,
            )
            await self._maybe_rediscover()
            await asyncio.sleep(_SERVER_WAIT_POLL_SEC)
