"""Base classes for GPU and CPU tools.

Every tool subclass must define a ``TOOL_PROMPT_DESCRIPTION`` class attribute
(a multi-line string) that documents the tool for the LLM system prompt.
The ``get_prompt_description()`` classmethod returns it, and the system prompt
builder aggregates all descriptions at runtime.
"""

import json
import logging
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Optional

import requests

from spatial_agent.gpu_rpc import (
    PROTOCOL_VERSION,
    RPCProtocolError,
    decode_response,
    encode_request,
)

logger = logging.getLogger(__name__)


def ensure_image_list(images) -> list:
    """Accept a single image or a list of images, always return a list."""
    if isinstance(images, (list, tuple)):
        return images
    from PIL import Image
    from spatial_agent.kernel_types.frame_image import FrameImage
    if isinstance(images, (Image.Image, FrameImage)):
        return [images]
    raise TypeError(
        f"Expected an image or list of images, got {type(images).__name__}."
    )


# ---------------------------------------------------------------------------
# GPU server registry (gpu_server.json)
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent.parent / "logs" / "gpu_server.json"

_COMMON_LIFECYCLE_ENV_KEYS = (
    "SPATIAL_AGENT_LEASE_ID",
    "SPATIAL_AGENT_COHORT_ID",
    "SPATIAL_AGENT_OWNER_ID",
    "SPATIAL_AGENT_WORKLOAD_POLICY_DIGEST",
    "SPATIAL_AGENT_SERVICE_MANIFEST_DIGEST",
)
_TOOL_LIFECYCLE_ENV_KEYS = _COMMON_LIFECYCLE_ENV_KEYS + (
    "SPATIAL_AGENT_TOOL_POOL_ID",
    "SPATIAL_AGENT_TOOL_NAMESPACE",
    "SPATIAL_AGENT_TOOL_SPEC_DIGEST",
    "SPATIAL_AGENT_TOOL_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST",
)
_VLLM_LIFECYCLE_ENV_KEYS = (
    "SPATIAL_AGENT_VLLM_POOL_ID",
    "SPATIAL_AGENT_VLLM_NAMESPACE",
    "SPATIAL_AGENT_VLLM_SERVER_SPEC_DIGEST",
    "SPATIAL_AGENT_VLLM_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST",
)
_GLOBAL_LIFECYCLE_ENV_KEYS = (
    _TOOL_LIFECYCLE_ENV_KEYS + _VLLM_LIFECYCLE_ENV_KEYS
)


def _tool_lifecycle_requested() -> bool:
    """Treat any core key, including cross-service residue, as managed."""
    return any(name in os.environ for name in _GLOBAL_LIFECYCLE_ENV_KEYS)


def _managed_tool_envelope() -> Optional[dict[str, Any]]:
    """Validate and return the exact managed-client GPU-tool envelope."""
    values = {
        name: os.getenv(name, "").strip()
        for name in _TOOL_LIFECYCLE_ENV_KEYS
        if name != "SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST"
    }
    missing = sorted(name for name, value in values.items() if not value)
    raw_slots = os.getenv("SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST", "")
    try:
        decoded_slots = json.loads(raw_slots)
    except (json.JSONDecodeError, TypeError):
        decoded_slots = None
    if (
        not isinstance(decoded_slots, list)
        or not decoded_slots
        or any(not isinstance(slot, str) or not slot.strip() for slot in decoded_slots)
    ):
        missing.append("SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST")
        slots: set[str] = set()
    else:
        slots = {slot.strip() for slot in decoded_slots}
        if len(slots) != len(decoded_slots):
            missing.append("SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST")

    if missing:
        logger.error(
            "Managed GPU-tool discovery has an incomplete or malformed exact "
            "client envelope (%s); failing closed",
            ", ".join(sorted(set(missing))),
        )
        return None
    return {**values, "slots": slots}


def _json_env_set(name: str) -> Optional[set[str]]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Malformed %s; failing managed discovery closed", name)
        return set()
    if not isinstance(value, list):
        logger.warning("Non-list %s; failing managed discovery closed", name)
        return set()
    return {str(item) for item in value}


def _managed_tool_record_is_attested(
    info: dict,
    *,
    expected_tool_spec_digest: str,
    expected_topology_digest: str,
) -> bool:
    """Validate one managed tool record against its frozen manifest identity."""
    if (
        not expected_tool_spec_digest
        or not expected_topology_digest
        or info.get("tool_spec_digest") != expected_tool_spec_digest
        or info.get("topology_digest") != expected_topology_digest
        or info.get("service") != "spatial-gpu-tools"
    ):
        return False

    backend = info.get("reconstruct_backend")
    if not backend or info.get("backend") != backend:
        return False
    if info.get("protocol_version") != PROTOCOL_VERSION:
        return False

    tools = info.get("tools")
    health = info.get("health")
    if not isinstance(tools, list) or not tools or not isinstance(health, dict):
        return False
    protocol = health.get("protocol")
    if not isinstance(protocol, dict) or (
        health.get("status") != "ok"
        or health.get("service") != "spatial-gpu-tools"
        or health.get("backend") != backend
        or protocol.get("name") != "spatial-gpu-rpc"
        or protocol.get("version") != PROTOCOL_VERSION
        or list(health.get("expected_tools") or []) != tools
        or list(health.get("deployed_tools") or []) != tools
    ):
        return False

    deployments = {f"spatial_{name}" for name in tools}
    if set(health.get("tools") or []) != deployments:
        return False
    effective_models = health.get("effective_models")
    if (
        not isinstance(effective_models, dict)
        or set(effective_models) != deployments
        or any(not isinstance(value, dict) for value in effective_models.values())
    ):
        return False

    gpu = health.get("gpu")
    try:
        if (
            not isinstance(gpu, dict)
            or int(gpu.get("requested_count")) != int(info.get("num_gpus"))
        ):
            return False
    except (TypeError, ValueError):
        return False

    keepalive = health.get("synthetic_keepalive")
    if (
        "synthetic_keepalive_requested" not in info
        or "synthetic_keepalive_effective" not in info
        or not isinstance(keepalive, dict)
        or "requested" not in keepalive
        or "effective" not in keepalive
        or (
            bool(keepalive.get("requested"))
            != bool(info.get("synthetic_keepalive_requested"))
            or bool(keepalive.get("effective"))
            != bool(info.get("synthetic_keepalive_effective"))
        )
    ):
        return False
    return True


def _read_registry() -> dict:
    """Read gpu_server.json and return the full dict."""
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        with open(_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if _tool_lifecycle_requested():
            raise RuntimeError(
                f"Managed GPU registry is unreadable or corrupt: {_REGISTRY_PATH}"
            ) from exc
        return {}
    if not isinstance(registry, dict):
        if _tool_lifecycle_requested():
            raise RuntimeError(
                f"Managed GPU registry root is not an object: {_REGISTRY_PATH}"
            )
        return {}
    return registry


def _is_port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _managed_tool_health_is_attested(
    info: dict, *, ip: str, port: int, auth_token: str,
) -> bool:
    """Revalidate one managed server's live authenticated identity."""
    url_host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    try:
        response = requests.get(
            f"http://{url_host}:{port}/health",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=5.0,
        )
        if response.status_code != 200:
            return False
        health = response.json()
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return False
    if not isinstance(health, dict):
        return False

    recorded_health = info.get("health")
    if not isinstance(recorded_health, dict):
        return False
    protocol = health.get("protocol")
    tools = info.get("tools")
    deployments = sorted(f"spatial_{name}" for name in tools or [])
    if (
        health.get("status") != "ok"
        or health.get("service") != "spatial-gpu-tools"
        or health.get("backend") != info.get("reconstruct_backend")
        or health.get("state") != "ready"
        or health.get("accepting") is not True
        or health.get("draining") is not False
        or protocol != {
            "name": "spatial-gpu-rpc", "version": PROTOCOL_VERSION,
        }
        or health.get("expected_tools") != tools
        or health.get("deployed_tools") != tools
        or sorted(health.get("tools") or []) != deployments
    ):
        return False

    for key in (
        "pool_id", "namespace", "slot_id", "tool_spec_digest",
        "topology_digest",
    ):
        if health.get(key) != info.get(key):
            return False
    if str(health.get("generation")) != str(info.get("generation")):
        return False

    for key in ("effective_models", "gpu", "synthetic_keepalive"):
        if health.get(key) != recorded_health.get(key):
            return False
    return True


def _deployment_to_tool(deployment_name: str) -> Optional[str]:
    """Map a client deployment name (e.g. ``spatial_Reconstruct``) to the short
    tool name stored in the registry (``Reconstruct``). Returns None if the
    deployment name is empty or does not follow the ``spatial_*`` convention."""
    if not deployment_name:
        return None
    prefix = "spatial_"
    return deployment_name[len(prefix):] if deployment_name.startswith(prefix) else deployment_name


def _find_alive_server_connection(
    registry: dict, tool_name: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Pick an authenticated server URL/token pair hosting ``tool_name``.

    ``tool_name`` is the short name stored in each registry entry's ``tools``
    list (e.g. ``"Reconstruct"`` or ``"SAM3"``). When None, any alive server
    is acceptable.
    """
    managed = _tool_lifecycle_requested()
    managed_envelope = _managed_tool_envelope() if managed else None
    if managed and managed_envelope is None:
        return None
    if not managed:
        # Preserve the historical flat, per-call randomized legacy routing.
        # Managed generations are excluded because an unleased client cannot
        # safely keep them alive during automatic drain.
        entries = [
            info for info in registry.values()
            if isinstance(info, dict) and not info.get("pool_id")
        ]
        random.shuffle(entries)
        for info in entries:
            if tool_name is not None and tool_name not in info.get("tools", []):
                continue
            ip = info.get("ip")
            port = info.get("http_port")
            auth_token = info.get("auth_token")
            if (
                not ip
                or not port
                or info.get("protocol_version") != PROTOCOL_VERSION
                or not isinstance(auth_token, str)
                or not auth_token
            ):
                continue
            if _is_port_open(ip, port):
                url_host = (
                    f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
                )
                return f"http://{url_host}:{port}", auth_token
            logger.debug(
                "[GPUTool] Server %s:%s unreachable, skipping.", ip, port,
            )
        return None
    uid_allowlist = _json_env_set("SPATIAL_AGENT_GPU_SERVER_UID_ALLOWLIST")
    slot_allowlist = (
        managed_envelope["slots"]
        if managed_envelope is not None
        else _json_env_set("SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST")
    )
    required_pool = (
        managed_envelope["SPATIAL_AGENT_TOOL_POOL_ID"]
        if managed_envelope is not None else ""
    )
    required_namespace = (
        managed_envelope["SPATIAL_AGENT_TOOL_NAMESPACE"]
        if managed_envelope is not None else ""
    )
    expected_tool_spec_digest = (
        managed_envelope["SPATIAL_AGENT_TOOL_SPEC_DIGEST"]
        if managed_envelope is not None else ""
    )
    expected_topology_digest = (
        managed_envelope["SPATIAL_AGENT_TOOL_TOPOLOGY_DIGEST"]
        if managed_envelope is not None else ""
    )
    candidates_by_slot: dict[str, list[tuple[int, str, dict]]] = {}
    for uid, info in registry.items():
        if not isinstance(info, dict):
            continue
        # Managed pools are lease-scoped.  A direct/unmanaged client has no
        # ownership record, so allowing it to use one would make automatic
        # drain unsafe.  Such clients must use a manual legacy pool or be
        # launched with the manager-provided pool/namespace manifest.
        if info.get("pool_id") and not required_pool:
            continue
        if uid_allowlist is not None and uid not in uid_allowlist:
            continue
        if required_pool and info.get("pool_id") != required_pool:
            continue
        if required_namespace and info.get("namespace") != required_namespace:
            continue
        if required_pool:
            try:
                attested = _managed_tool_record_is_attested(
                    info,
                    expected_tool_spec_digest=expected_tool_spec_digest,
                    expected_topology_digest=expected_topology_digest,
                )
            except (TypeError, ValueError):
                # One malformed registry record must not hide an otherwise
                # valid generation for the same frozen slot.
                attested = False
            if not attested:
                continue
        # A successor is not eligible to replace its predecessor until it is
        # fully accepting.  Grouping first would let a higher STARTING
        # generation hide the still-healthy prior generation for this slot.
        if info.get("state", "ready") not in {"ready", "accepting"}:
            continue
        if info.get("draining") or info.get("accepting") is False:
            continue
        slot_id = str(info.get("slot_id") or uid)
        if slot_allowlist is not None and slot_id not in slot_allowlist:
            continue
        try:
            generation = int(info.get("generation") or 0)
        except (TypeError, ValueError):
            continue
        candidates_by_slot.setdefault(slot_id, []).append(
            (generation, str(uid), info)
        )

    # Resolve health *within* each logical slot.  Selecting the newest record
    # before validating its connection would let a dead or malformed
    # successor shadow a healthy predecessor.  UID allowlists, when present,
    # have already been applied above, so an exact-generation pin still fails
    # closed instead of falling back outside its authority.
    reachable_slots: set[str] = set()
    entries: list[tuple[str, dict]] = []
    for slot_id, candidates in candidates_by_slot.items():
        selected: Optional[tuple[str, dict]] = None
        for _generation, uid, info in sorted(
            candidates, key=lambda item: (item[0], item[1]), reverse=True
        ):
            tools = info.get("tools")
            ip = info.get("ip")
            port = info.get("http_port")
            auth_token = info.get("auth_token")
            try:
                port_number = int(port)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(tools, list)
                or not tools
                or not isinstance(ip, str)
                or not ip
                or not 1 <= port_number <= 65535
                or info.get("protocol_version") != PROTOCOL_VERSION
                or not isinstance(auth_token, str)
                or not auth_token
            ):
                continue
            healthy = _managed_tool_health_is_attested(
                info,
                ip=ip,
                port=port_number,
                auth_token=auth_token,
            )
            if not healthy:
                logger.debug(
                    "[GPUTool] Server %s:%s unhealthy, trying prior "
                    "generation for slot %s.",
                    ip,
                    port_number,
                    slot_id,
                )
                continue
            reachable_slots.add(slot_id)
            if tool_name is None or tool_name in tools:
                selected = (uid, {**info, "http_port": port_number})
                break
        if selected is not None:
            entries.append(selected)

    if slot_allowlist is not None and reachable_slots != slot_allowlist:
        return None
    random.shuffle(entries)
    for uid, info in entries:
        ip = info.get("ip")
        port = info.get("http_port")
        auth_token = info.get("auth_token")
        url_host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
        return f"http://{url_host}:{port}", auth_token
    return None


def _find_alive_server(registry: dict, tool_name: Optional[str] = None) -> Optional[str]:
    """Return only the URL of a compatible server, for availability checks."""
    connection = _find_alive_server_connection(registry, tool_name)
    return connection[0] if connection else None


def _get_server_connection(tool_name: Optional[str] = None) -> tuple[str, str]:
    """Get an alive authenticated GPU server connection for ``tool_name``.

    Non-sticky: selection runs per call so load spreads across all eligible
    servers. Waits up to 4 hours if none are available, polling every 10s.
    """
    registry = _read_registry()
    connection = _find_alive_server_connection(registry, tool_name)

    if connection is None:
        logger.debug("[GPUTool] No alive GPU server found for tool=%s. Waiting...", tool_name)
        for attempt in range(1440):  # 1440 x 10s = 4 hours
            time.sleep(10)
            registry = _read_registry()
            connection = _find_alive_server_connection(registry, tool_name)
            if connection is not None:
                break
            if (attempt + 1) % 6 == 0:
                elapsed_min = (attempt + 1) * 10 // 60
                logger.debug("[GPUTool] Still waiting for GPU server... (%dm)", elapsed_min)
        else:
            raise RuntimeError(
                "No alive GPU server found in gpu_server.json after 4 hours. "
                "Start a GPU server via the agent manager."
            )

    url, auth_token = connection
    logger.debug("[GPUTool] Using GPU server at %s (tool=%s)", url, tool_name)
    return url, auth_token


def _get_server_url(tool_name: Optional[str] = None) -> str:
    """Get only the URL of an alive authenticated GPU server."""
    return _get_server_connection(tool_name)[0]


# ---------------------------------------------------------------------------
# HTTP request timeout (per-request, not connection)
# Kept strictly below the Jupyter cell timeout (config.timeout_sec = 600s) so
# that a stalled GPU request fails HTTP-timeout with budget left for
# ``_call_remote`` to retry on a different server via per-call selection.
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 450  # seconds


def _resolve_tool_prompt(cls, ablations: dict = None) -> str:
    """Resolve a tool's prompt description, applying sub-section ablations if defined.

    If the tool defines ``TOOL_PROMPT_SECTIONS`` (OrderedDict of {sub_name: content}),
    each sub-section is resolved through ``resolve_section()`` using keys like
    ``tool_sam3_api``, ``tool_sam3_verify``, etc.  The prefix comes from
    ``TOOL_ABLATION_PREFIX`` (e.g., ``"tool_sam3"``).

    Whole-tool exclusion is also supported: excluding ``"tool_sam3"`` removes the
    entire description.

    Tools without ``TOOL_PROMPT_SECTIONS`` fall back to the monolithic
    ``TOOL_PROMPT_DESCRIPTION`` string (unchanged behavior).
    """
    from spatial_agent.llm.prompt_common import resolve_section

    sections = getattr(cls, "TOOL_PROMPT_SECTIONS", None)
    if sections is None:
        # Legacy: single monolithic string, no sub-section ablation
        return cls.TOOL_PROMPT_DESCRIPTION.strip()

    prefix = getattr(cls, "TOOL_ABLATION_PREFIX", "tool_unknown")

    # Whole-tool exclusion
    if ablations and prefix in ablations.get("exclude", []):
        logger.info("[prompt ablation] EXCLUDED (whole tool): %s", prefix)
        return ""

    # Whole-tool override
    if ablations:
        override_path = ablations.get("override", {}).get(prefix)
        if override_path:
            logger.info("[prompt ablation] OVERRIDDEN (whole tool): %s -> %s", prefix, override_path)
            with open(override_path, "r") as f:
                return f.read().strip()

    # Resolve each sub-section
    parts = []
    for sub_name, default_content in sections.items():
        key = f"{prefix}_{sub_name}"
        resolved = resolve_section(key, default_content.strip(), ablations or {})
        if resolved:
            parts.append(resolved.strip())

    return "\n\n".join(parts)


def get_all_tool_ablation_names(*tool_classes) -> set:
    """Collect all valid ablation section names from the given tool classes."""
    names = set()
    for cls in tool_classes:
        sections = getattr(cls, "TOOL_PROMPT_SECTIONS", None)
        if sections is None:
            continue
        prefix = getattr(cls, "TOOL_ABLATION_PREFIX", "tool_unknown")
        names.add(prefix)  # whole-tool key
        for sub_name in sections:
            names.add(f"{prefix}_{sub_name}")
    return names


class GPUTool:
    """Base class for tools that communicate with the GPU server over HTTP.

    Subclasses provide synchronous methods that the Jupyter kernel calls.
    Each remote call is an authenticated HTTP POST with safely encoded JSON data.
    """

    TOOL_PROMPT_DESCRIPTION: str = ""  # Override in subclasses

    def __init__(self, deployment_name: str = "", gpu_tool_max_retries: int = 3):
        self._deployment_name = deployment_name
        self._max_retries = max(gpu_tool_max_retries, 1)

    @classmethod
    def get_prompt_description(cls, ablations: dict = None) -> str:
        """Return the prompt description, resolving sub-section ablations if defined."""
        return _resolve_tool_prompt(cls, ablations)

    @property
    def is_available(self) -> bool:
        """True if any GPU server hosting this deployment is reachable."""
        if self._deployment_name:
            try:
                _get_server_url(_deployment_to_tool(self._deployment_name))
                return True
            except RuntimeError:
                return False
        return False

    def _call_remote(self, method_name: str, **kwargs) -> Any:
        """Call a remote method on the GPU server via HTTP POST.

        The request body is JSON: {version, deployment, method, kwargs}.
        Images, arrays, and known result dataclasses use explicit tagged values.

        Retries up to ``self._max_retries`` times with exponential backoff.
        Each attempt re-runs server selection, so a retry naturally lands on
        a different server when multiple are alive.
        """
        payload = encode_request(self._deployment_name, method_name, kwargs)
        tool_name = _deployment_to_tool(self._deployment_name)

        for attempt in range(self._max_retries):
            try:
                url, auth_token = _get_server_connection(tool_name)
                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }
                # Attribution is carried only in HTTP headers; it never enters
                # the tool payload seen by the model or tool implementation.
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
                        headers[header] = value

                resp = requests.post(
                    f"{url}/call",
                    data=payload,
                    headers=headers,
                    timeout=_HTTP_TIMEOUT,
                )

                result = decode_response(resp.content)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"GPU server returned HTTP {resp.status_code} without an error"
                    )
                return result

            except Exception as exc:
                if _is_application_error(exc):
                    raise

                if attempt < self._max_retries - 1:
                    backoff = 5 * (2 ** attempt)
                    logger.warning(
                        "[GPUTool] %s.%s attempt %d/%d failed: %s. "
                        "Retrying in %ds...",
                        self.__class__.__name__, method_name,
                        attempt + 1, self._max_retries, exc, backoff,
                    )
                    time.sleep(backoff)

        raise RuntimeError(
            f"{self.__class__.__name__}.{method_name} is temporarily unavailable. "
            f"The GPU server may be restarting. Please try again."
        )


def _is_application_error(exc: Exception) -> bool:
    """True if the exception is an application-level error (not infra).

    Application errors (bad inputs, assertion failures) should not be retried.
    """
    if isinstance(exc, RPCProtocolError):
        return False
    if isinstance(exc, (AssertionError, ValueError, TypeError)):
        return True
    return False


class CPUTool:
    """Base class for CPU-only tools that run directly in-process."""

    TOOL_PROMPT_DESCRIPTION: str = ""  # Override in subclasses

    @classmethod
    def get_prompt_description(cls, ablations: dict = None) -> str:
        """Return the prompt description, resolving sub-section ablations if defined."""
        return _resolve_tool_prompt(cls, ablations)
