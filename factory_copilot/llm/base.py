from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "args": dict}] when native tool calling is used


class LLM(ABC):
    """Minimal chat interface.

    `json_mode` asks the provider for a JSON object when supported. `tools` (OpenAI function-schema list) enables native
    tool calling where the provider supports it; the response then carries `tool_calls`. Providers without native tool
    calling ignore `tools` and the caller falls back to parsing a JSON decision from `text`.
    """

    name: str = "llm"

    @abstractmethod
    def complete(self, messages: list[Message], *, max_tokens: int = 800, temperature: float = 0.0, json_mode: bool = False, tools: list[dict] | None = None) -> LLMResponse: ...
