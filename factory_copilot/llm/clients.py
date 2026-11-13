"""Provider clients. Both are thin HTTP wrappers (httpx) so there is no SDK lock-in and the request/response shapes
are visible in one place. Pricing is passed in, not hard-coded, so cost logs stay honest when prices change.
"""

from __future__ import annotations

import json
import time

import httpx

from factory_copilot.config import settings
from factory_copilot.llm.base import LLM, LLMResponse, Message

RETRY_STATUSES = {429, 500, 502, 503, 529}


def _post_with_retry(url: str, headers: dict, body: dict, timeout: float, attempts: int = 5) -> dict:
    """POST with exponential backoff on rate limits and transient server errors. Honours Retry-After when present."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
            if r.status_code in RETRY_STATUSES and attempt < attempts - 1:
                wait = float(r.headers.get("retry-after", delay))
                time.sleep(min(wait, 60.0))
                delay *= 2
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:400]}")
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"LLM request failed after {attempts} attempts: {last}")


class OpenAICompatibleLLM(LLM):
    """Works with OpenAI, Groq, Together, Ollama, vLLM: anything that serves /chat/completions."""

    def __init__(self, model: str, api_key: str, base_url: str, price_in: float, price_out: float, timeout: float = 60.0):
        self.name = model
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")
        self.price_in, self.price_out, self.timeout = price_in, price_out, timeout

    def complete(self, messages, *, max_tokens=800, temperature=0.0, json_mode=False, tools=None) -> LLMResponse:
        body = {"model": self.model, "messages": [{"role": m.role, "content": m.content} for m in messages], "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "required"
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            # Reasoning models spend output tokens thinking; keep it short for structured steps so JSON is never truncated.
            body["reasoning_effort"] = "low"
        t0 = time.perf_counter()
        data = _post_with_retry(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body, self.timeout)
        usage = data.get("usage", {})
        i, o = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": fn.get("name", ""), "args": args})
        return LLMResponse(
            text=msg.get("content") or "", model=self.model, input_tokens=i, output_tokens=o,
            cost_usd=(i * self.price_in + o * self.price_out) / 1e6,
            latency_s=time.perf_counter() - t0, raw={"id": data.get("id")}, tool_calls=calls,
        )


class AnthropicLLM(LLM):
    def __init__(self, model: str, api_key: str, price_in: float, price_out: float, timeout: float = 60.0):
        self.name = model
        self.model, self.api_key, self.price_in, self.price_out, self.timeout = model, api_key, price_in, price_out, timeout

    def complete(self, messages, *, max_tokens=800, temperature=0.0, json_mode=False, tools=None) -> LLMResponse:
        system = "\n".join(m.content for m in messages if m.role == "system")
        if json_mode and not tools:
            system += "\nRespond with a single JSON object and nothing else."
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                "messages": [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]}
        if tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""), "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})} for t in tools]
        t0 = time.perf_counter()
        data = _post_with_retry("https://api.anthropic.com/v1/messages", {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body, self.timeout)
        i, o = int(data["usage"]["input_tokens"]), int(data["usage"]["output_tokens"])
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        calls = [{"name": b["name"], "args": b.get("input", {})} for b in data.get("content", []) if b.get("type") == "tool_use"]
        return LLMResponse(text=text, model=self.model, input_tokens=i, output_tokens=o, cost_usd=(i * self.price_in + o * self.price_out) / 1e6, latency_s=time.perf_counter() - t0, raw={"id": data.get("id")}, tool_calls=calls)


class LLMConfigError(RuntimeError):
    pass


def _is_local(url: str) -> bool:
    return any(h in url for h in ("://localhost", "://127.0.0.1", "://0.0.0.0", "://host.docker.internal", "://ollama"))


def make_llm(tier: str) -> LLM:
    """tier: 'cheap' or 'strong'. The mock is used only when LLM_PROVIDER=mock is chosen explicitly.

    A real provider with no key fails loudly instead of silently becoming the mock, so a demo or an evaluation
    cannot look real while running the deterministic fixture. Local OpenAI-compatible servers (Ollama, vLLM on
    localhost) may run without a key.
    """
    from factory_copilot.llm.mock import MockLLM

    if settings.llm_provider == "mock":
        return MockLLM(name=f"mock-{tier}")
    if not settings.llm_api_key and not (settings.llm_provider == "openai_compatible" and _is_local(settings.llm_base_url)):
        raise LLMConfigError(f"LLM_PROVIDER={settings.llm_provider} but LLM_API_KEY is empty. Set the key, or set LLM_PROVIDER=mock to use the offline mock on purpose.")
    model = settings.cheap_model if tier == "cheap" else settings.strong_model
    pin = settings.cheap_price_in if tier == "cheap" else settings.strong_price_in
    pout = settings.cheap_price_out if tier == "cheap" else settings.strong_price_out
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(model, settings.llm_api_key, pin, pout, settings.request_timeout_s)
    return OpenAICompatibleLLM(model, settings.llm_api_key, settings.llm_base_url, pin, pout, settings.request_timeout_s)
