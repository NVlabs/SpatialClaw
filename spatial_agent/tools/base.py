"""Base classes for GPU and CPU tools.

Every tool subclass must define a ``TOOL_PROMPT_DESCRIPTION`` class attribute
(a multi-line string) that documents the tool for the LLM system prompt.
The ``get_prompt_description()`` classmethod returns it, and the system prompt
builder aggregates all descriptions at runtime.
"""

import json
import logging
import pickle
import random
import socket
import time
from pathlib import Path
from typing import Any, Optional

import requests

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


def _read_registry() -> dict:
    """Read gpu_server.json and return the full dict."""
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        with open(_REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _is_port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _deployment_to_tool(deployment_name: str) -> Optional[str]:
    """Map a client deployment name (e.g. ``spatial_Reconstruct``) to the short
    tool name stored in the registry (``Reconstruct``). Returns None if the
    deployment name is empty or does not follow the ``spatial_*`` convention."""
    if not deployment_name:
        return None
    prefix = "spatial_"
    return deployment_name[len(prefix):] if deployment_name.startswith(prefix) else deployment_name


def _find_alive_server(registry: dict, tool_name: Optional[str] = None) -> Optional[str]:
    """Pick a random alive server URL that hosts ``tool_name``.

    ``tool_name`` is the short name stored in each registry entry's ``tools``
    list (e.g. ``"Reconstruct"`` or ``"SAM3"``). When None, any alive server
    is acceptable.
    """
    entries = list(registry.values())
    random.shuffle(entries)
    for info in entries:
        if tool_name is not None and tool_name not in info.get("tools", []):
            continue
        ip = info.get("ip")
        port = info.get("http_port")
        if not ip or not port:
            continue
        if _is_port_open(ip, port):
            return f"http://{ip}:{port}"
        else:
            logger.debug("[GPUTool] Server %s:%s unreachable, skipping.", ip, port)
    return None


def _get_server_url(tool_name: Optional[str] = None) -> str:
    """Get the HTTP URL of an alive GPU server hosting ``tool_name``.

    Non-sticky: selection runs per call so load spreads across all eligible
    servers. Waits up to 4 hours if none are available, polling every 10s.
    """
    registry = _read_registry()
    url = _find_alive_server(registry, tool_name)

    if url is None:
        logger.debug("[GPUTool] No alive GPU server found for tool=%s. Waiting...", tool_name)
        for attempt in range(1440):  # 1440 x 10s = 4 hours
            time.sleep(10)
            registry = _read_registry()
            url = _find_alive_server(registry, tool_name)
            if url is not None:
                break
            if (attempt + 1) % 6 == 0:
                elapsed_min = (attempt + 1) * 10 // 60
                logger.debug("[GPUTool] Still waiting for GPU server... (%dm)", elapsed_min)
        else:
            raise RuntimeError(
                "No alive GPU server found in gpu_server.json after 4 hours. "
                "Start a GPU server via the agent manager."
            )

    logger.debug("[GPUTool] Using GPU server at %s (tool=%s)", url, tool_name)
    return url


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
    Each remote call is a simple HTTP POST with pickle-serialized data.
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

        The request body is pickle-serialized: {deployment, method, kwargs}.
        The response body is pickle-serialized: the return value or exception.

        Retries up to ``self._max_retries`` times with exponential backoff.
        Each attempt re-runs server selection, so a retry naturally lands on
        a different server when multiple are alive.
        """
        payload = pickle.dumps({
            "deployment": self._deployment_name,
            "method": method_name,
            "kwargs": kwargs,
        })
        tool_name = _deployment_to_tool(self._deployment_name)

        for attempt in range(self._max_retries):
            try:
                url = _get_server_url(tool_name)
                resp = requests.post(
                    f"{url}/call",
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=_HTTP_TIMEOUT,
                )

                result = pickle.loads(resp.content)

                if resp.status_code == 200:
                    return result

                # Server returned an error — result is the exception
                if isinstance(result, Exception):
                    raise result
                raise RuntimeError(f"GPU server error (HTTP {resp.status_code}): {result}")

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
