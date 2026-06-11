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
import io
import json
import logging
import os
import random
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
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
        self._endpoints: List[str] = []
        self._api_key = (
            config.llm_api_key
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "bearer"
        )
        self._is_vllm = config.llm_base_url.lower().strip() == "vllm"

        # Client pool: endpoint URL -> AsyncOpenAI
        self._client_pool: Dict[str, AsyncOpenAI] = {}

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
        # threading.Lock is not picklable — recreate on the other side.
        state["_usage_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._usage_lock is None:
            self._usage_lock = threading.Lock()

    # ------------------------------------------------------------------
    # vLLM discovery
    # ------------------------------------------------------------------

    def _discover_vllm_endpoints(self) -> List[str]:
        """Read logs/serve.json and return all endpoints for the model."""
        serve_file = os.path.join(
            os.path.dirname(__file__), "..", "logs", "serve.json"
        )
        serve_file = os.path.abspath(serve_file)
        if not os.path.exists(serve_file):
            return []
        try:
            with open(serve_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        model_instances = data.get(self._model, {})
        endpoints = []
        for instance in model_instances.values():
            ip = instance["ip"]
            port = instance["port"]
            endpoints.append(f"http://{ip}:{port}/v1")
        return sorted(endpoints)  # sorted for stable sticky routing

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

    def _health_check_endpoints(self) -> None:
        """Probe each endpoint at startup, drop unreachable ones."""
        if not self._is_vllm or not self._endpoints:
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
        if now - self._last_discovery < _DISCOVERY_TTL_SEC:
            return

        # Lazy init lock (must be created in async context)
        if self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()

        async with self._discovery_lock:
            # Double-check after acquiring lock
            if time.monotonic() - self._last_discovery < _DISCOVERY_TTL_SEC:
                return

            new_endpoints = self._discover_vllm_endpoints()
            if not new_endpoints:
                self._last_discovery = time.monotonic()
                return

            if new_endpoints != self._endpoints:
                # Close clients for removed endpoints
                removed = set(self._endpoints) - set(new_endpoints)
                for ep in removed:
                    client = self._client_pool.pop(ep, None)
                    if client:
                        try:
                            await client.close()
                        except Exception:
                            pass
                    self._unhealthy.pop(ep, None)
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
        return healthy if healthy else self._endpoints

    def _pick_endpoint(self, session_id: Optional[str] = None) -> str:
        """Pick endpoint using least-connections, with optional sticky preference."""
        candidates = self._get_healthy_endpoints()
        if not candidates:
            candidates = self._endpoints

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

    def _track_request_start(self, endpoint: str) -> None:
        """Increment active request counter for an endpoint."""
        self._active_requests[endpoint] += 1

    def _track_request_end(self, endpoint: str) -> None:
        """Decrement active request counter for an endpoint."""
        self._active_requests[endpoint] = max(0, self._active_requests[endpoint] - 1)

    # ------------------------------------------------------------------
    # Client pool
    # ------------------------------------------------------------------

    def _get_client(self, endpoint: str) -> AsyncOpenAI:
        """Get or create a pooled AsyncOpenAI client for the endpoint."""
        client = self._client_pool.get(endpoint)
        if client is None:
            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=endpoint,
            )
            self._client_pool[endpoint] = client
        return client

    async def close(self) -> None:
        """Close all pooled clients. Call on shutdown."""
        for ep, client in self._client_pool.items():
            try:
                await client.close()
            except Exception:
                pass
        self._client_pool.clear()

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
        Standard OpenAI params if set: top_p, presence_penalty.
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

        # Proactive re-discovery before picking endpoint
        await self._maybe_rediscover()

        wait_start: Optional[float] = None

        while True:
            last_exc: Optional[Exception] = None
            is_connection_failure = False

            for attempt in range(max_retries):
                endpoint = self._pick_endpoint(session_id)
                client = self._get_client(endpoint)
                self._track_request_start(endpoint)
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            **api_kwargs,
                        ),
                        timeout=600,
                    )
                except self._RETRYABLE_ERRORS as exc:
                    last_exc = exc
                    is_connection_failure = isinstance(exc, self._CONNECTION_ERRORS)
                    if is_connection_failure:
                        self._mark_unhealthy(endpoint)
                    logger.warning(
                        "[LLMClient] Attempt %d/%d failed (%s: %s), retrying...",
                        attempt + 1, max_retries, type(exc).__name__, exc,
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
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    raise
                finally:
                    self._track_request_end(endpoint)

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

        # Build multimodal message
        user_content: List[Dict[str, Any]] = []
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_to_base64_url(img)},
            })
        user_content.append({"type": "text", "text": question})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # Proactive re-discovery before picking endpoint
        await self._maybe_rediscover()

        max_retries = 5
        wait_start: Optional[float] = None

        while True:
            last_exc: Optional[Exception] = None
            is_connection_failure = False

            for attempt in range(max_retries):
                endpoint = self._pick_endpoint(session_id)
                client = self._get_client(endpoint)
                self._track_request_start(endpoint)
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            **api_kwargs,
                        ),
                        timeout=self.config.vlm_query_timeout_sec,
                    )
                except self._RETRYABLE_ERRORS as exc:
                    last_exc = exc
                    is_connection_failure = isinstance(exc, self._CONNECTION_ERRORS)
                    if is_connection_failure:
                        self._mark_unhealthy(endpoint)
                    logger.warning(
                        "[LLMClient] VLM attempt %d/%d failed (%s: %s), retrying...",
                        attempt + 1, max_retries, type(exc).__name__, exc,
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
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    raise
                finally:
                    self._track_request_end(endpoint)

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
