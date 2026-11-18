"""Core Pydantic v2 data structures for the Zero-Framework Async ReAct Runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
RiskLevel = Literal["read", "write"]


class ToolCallFunction(BaseModel):
    """Raw function call emitted by an OpenAI-compatible chat model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    """Tool invocation descriptor inside an assistant message."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    """Canonical conversation message compatible with OpenAI Chat Completions wire format."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            payload["content"] = self.content
            payload["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        else:
            payload["content"] = self.content if self.content is not None else ""
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None and self.role == "tool":
            payload["name"] = self.name
        return payload


class LLMUsage(BaseModel):
    """Token consumption metrics for a single LLM completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalized response returned by both OpenAI HTTP and Mock LLM clients."""

    message: Message
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = "mock-react-v1"
    latency_ms: float = 0.0
    retries: int = 0


class RuntimeConfig(BaseModel):
    """Execution policies and budgets for the ReAct loop."""

    max_steps: int = Field(default=10, ge=1, le=50)
    max_context_tokens: int = Field(default=3200, ge=256)
    keep_recent_groups: int = Field(default=3, ge=1)
    max_self_heal_retries: int = Field(default=3, ge=0)
    enable_self_healing: bool = True
    enable_compression: bool = True
    parallel_tool_calls: bool = True
    stream: bool = False


class StepTrace(BaseModel):
    """Observability record for one ReAct reasoning/acting step."""

    step: int
    thought: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    self_healed_errors: int = 0
    security_blocks: int = 0
    compressed_before_step: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class AgentRunResult(BaseModel):
    """Structured final result produced by ReActRuntime.run()."""

    final_answer: str
    status: Literal["completed", "max_steps_exceeded", "aborted"]
    steps: int
    messages: list[Message]
    traces: list[StepTrace] = Field(default_factory=list)
    self_heal_count: int = 0
    compression_count: int = 0
    blocked_security_calls: int = 0
    cost_summary: dict[str, Any] = Field(default_factory=dict)
