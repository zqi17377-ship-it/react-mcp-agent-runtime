"""Self-Healing, Parallel Async ReAct Agent State Loop (`ReActRuntime`)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from react_mcp_runtime.context import SlidingWindowCompressor, validate_tool_call_integrity
from react_mcp_runtime.cost_tracker import CostTracker
from react_mcp_runtime.llm_client import BaseLLMClient, ContextLengthExceededError
from react_mcp_runtime.schema import (
    AgentRunResult,
    Message,
    RuntimeConfig,
    StepTrace,
    ToolCall,
)
from react_mcp_runtime.tools import ToolRegistry, ToolSecurityBlockedError

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous reliability & software diagnostics ReAct agent. "
    "Use registered tools methodically, prefer parallel read-only queries when independent, "
    "respect security/HITL boundaries, and ground every conclusion in verified tool outputs."
)


class MalformedToolCallError(ValueError):
    """Raised when tool-call JSON or schema validation fails and self-healing is disabled/exhausted."""


@dataclass
class _SingleToolOutcome:
    message: Message
    self_heal_error: bool = False
    security_blocked: bool = False
    fatal_error: str | None = None


class ReActRuntime:
    """Zero-framework async ReAct loop with parallel tools, JSON/schema self-healing & compression."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        config: RuntimeConfig | None = None,
        cost_tracker: CostTracker | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_step: Callable[[StepTrace, Message], None] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config or RuntimeConfig()
        self.cost_tracker = cost_tracker or CostTracker()
        self.system_prompt = system_prompt
        self.on_step = on_step
        self.compressor = SlidingWindowCompressor(
            max_context_tokens=self.config.max_context_tokens,
            keep_recent_groups=self.config.keep_recent_groups,
        )

    async def run(
        self,
        user_prompt: str,
        *,
        extra_messages: list[Message] | None = None,
    ) -> AgentRunResult:
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append(Message(role="user", content=user_prompt))

        traces: list[StepTrace] = []
        self_heal_count = 0
        compression_count = 0
        blocked_security_calls = 0
        tool_schemas = self.tool_registry.openai_schemas()
        schema_overhead = max(0, len(str(tool_schemas)) // 4) if tool_schemas else 0

        for step_idx in range(1, self.config.max_steps + 1):
            did_compress = False
            if self.config.enable_compression:
                messages, did_compress = await self.compressor.compress_if_needed(
                    messages,
                    overhead_tokens=schema_overhead,
                )
                if did_compress:
                    compression_count += 1

            validate_tool_call_integrity(messages)

            try:
                response = await self.llm_client.complete(
                    messages,
                    tools=tool_schemas if tool_schemas else None,
                    stream=self.config.stream,
                )
            except ContextLengthExceededError as exc:
                return AgentRunResult(
                    final_answer=f"Aborted [context_length_exceeded]: {exc}",
                    status="aborted",
                    steps=step_idx - 1,
                    messages=messages,
                    traces=traces,
                    self_heal_count=self_heal_count,
                    compression_count=compression_count,
                    blocked_security_calls=blocked_security_calls,
                    cost_summary=self.cost_tracker.summary_dict(),
                )

            self.cost_tracker.record(step_idx, response)
            assistant_msg = response.message
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                trace = StepTrace(
                    step=step_idx,
                    thought=assistant_msg.content,
                    tool_names=[],
                    compressed_before_step=did_compress,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=round(response.latency_ms, 2),
                )
                traces.append(trace)
                if self.on_step:
                    self.on_step(trace, assistant_msg)
                return AgentRunResult(
                    final_answer=assistant_msg.content or "",
                    status="completed",
                    steps=step_idx,
                    messages=messages,
                    traces=traces,
                    self_heal_count=self_heal_count,
                    compression_count=compression_count,
                    blocked_security_calls=blocked_security_calls,
                    cost_summary=self.cost_tracker.summary_dict(),
                )

            # Dispatch tool calls (parallel via asyncio.gather when enabled & >1 call)
            tcs = assistant_msg.tool_calls
            if self.config.parallel_tool_calls and len(tcs) > 1:
                outcomes = await asyncio.gather(
                    *(self._execute_one_tool_call(tc, self_heal_count) for tc in tcs)
                )
            else:
                outcomes = []
                for tc in tcs:
                    outcomes.append(
                        await self._execute_one_tool_call(
                            tc,
                            self_heal_count + sum(1 for o in outcomes if o.self_heal_error),
                        )
                    )

            step_heals = 0
            step_blocks = 0
            fatal_reasons: list[str] = []
            for outcome in outcomes:
                messages.append(outcome.message)
                if outcome.self_heal_error:
                    step_heals += 1
                if outcome.security_blocked:
                    step_blocks += 1
                if outcome.fatal_error:
                    fatal_reasons.append(outcome.fatal_error)

            blocked_security_calls += step_blocks

            trace = StepTrace(
                step=step_idx,
                thought=assistant_msg.content,
                tool_names=[tc.function.name for tc in tcs],
                self_healed_errors=step_heals if not fatal_reasons else 0,
                security_blocks=step_blocks,
                compressed_before_step=did_compress,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=round(response.latency_ms, 2),
            )
            traces.append(trace)
            if self.on_step:
                self.on_step(trace, assistant_msg)

            if fatal_reasons:
                return AgentRunResult(
                    final_answer=f"Aborted [tool_call_unrecoverable]: {'; '.join(fatal_reasons)}",
                    status="aborted",
                    steps=step_idx,
                    messages=messages,
                    traces=traces,
                    self_heal_count=self_heal_count,
                    compression_count=compression_count,
                    blocked_security_calls=blocked_security_calls,
                    cost_summary=self.cost_tracker.summary_dict(),
                )

            self_heal_count += step_heals

        return AgentRunResult(
            final_answer=f"Exceeded max_steps={self.config.max_steps} without reaching final synthesis.",
            status="max_steps_exceeded",
            steps=self.config.max_steps,
            messages=messages,
            traces=traces,
            self_heal_count=self_heal_count,
            compression_count=compression_count,
            blocked_security_calls=blocked_security_calls,
            cost_summary=self.cost_tracker.summary_dict(),
        )

    async def _execute_one_tool_call(
        self,
        tc: ToolCall,
        prior_self_heals: int,
    ) -> _SingleToolOutcome:
        tool_name = tc.function.name
        raw_args_str = tc.function.arguments

        # 1. JSON Parsing & Self-Healing
        try:
            parsed_args = json.loads(raw_args_str) if raw_args_str.strip() else {}
            if not isinstance(parsed_args, dict):
                raise json.JSONDecodeError("Tool arguments root must be a JSON object", raw_args_str, 0)
        except json.JSONDecodeError as exc:
            err_payload: dict[str, Any] = {
                "ok": False,
                "error_type": "json_decode_error",
                "tool": tool_name,
                "error": f"{exc.msg} at line {exc.lineno} col {exc.colno}",
                "raw_arguments": raw_args_str[:240],
                "repair_hint": "Re-emit the tool call with strictly valid RFC8259 JSON object syntax.",
            }
            msg = Message(
                role="tool",
                tool_call_id=tc.id,
                name=tool_name,
                content=json.dumps(err_payload, ensure_ascii=False),
            )
            if not self.config.enable_self_healing:
                return _SingleToolOutcome(
                    message=msg,
                    fatal_error=f"JSONDecodeError on `{tool_name}` (self-healing disabled)",
                )
            if prior_self_heals >= self.config.max_self_heal_retries:
                return _SingleToolOutcome(
                    message=msg,
                    fatal_error=f"Exceeded max_self_heal_retries={self.config.max_self_heal_retries} on `{tool_name}`",
                )
            return _SingleToolOutcome(message=msg, self_heal_error=True)

        # 2. Tool Lookup & Pydantic v2 Argument Validation
        spec = self.tool_registry.get(tool_name)
        if spec is None:
            err_payload = {
                "ok": False,
                "error_type": "validation_error",
                "tool": tool_name,
                "errors": [f"Unknown tool '{tool_name}'. Available tools: {self.tool_registry.names()}"],
                "repair_hint": "Select one of the registered tool names.",
            }
            msg = Message(
                role="tool",
                tool_call_id=tc.id,
                name=tool_name,
                content=json.dumps(err_payload, ensure_ascii=False),
            )
            if not self.config.enable_self_healing or prior_self_heals >= self.config.max_self_heal_retries:
                return _SingleToolOutcome(
                    message=msg,
                    fatal_error=f"Unknown tool `{tool_name}`",
                )
            return _SingleToolOutcome(message=msg, self_heal_error=True)

        try:
            output_str = await spec.invoke(parsed_args)
            # Check if an MCP server returned a structured security block inside tool result
            sec_blocked = False
            try:
                out_json = json.loads(output_str)
                if isinstance(out_json, dict) and out_json.get("error_type") == "security_blocked":
                    sec_blocked = True
            except Exception:
                pass
            return _SingleToolOutcome(
                message=Message(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tool_name,
                    content=output_str,
                ),
                security_blocked=sec_blocked,
            )
        except ValidationError as exc:
            compact_errors = [
                {
                    "loc": list(e.get("loc", ())),
                    "msg": e.get("msg", ""),
                    "type": e.get("type", ""),
                }
                for e in exc.errors()
            ]
            err_payload = {
                "ok": False,
                "error_type": "validation_error",
                "tool": tool_name,
                "errors": compact_errors,
                "expected_schema": spec.parameters_schema,
                "repair_hint": "Fix argument names, types, and enum values to match `expected_schema`.",
            }
            msg = Message(
                role="tool",
                tool_call_id=tc.id,
                name=tool_name,
                content=json.dumps(err_payload, ensure_ascii=False),
            )
            if not self.config.enable_self_healing:
                return _SingleToolOutcome(
                    message=msg,
                    fatal_error=f"ValidationError on `{tool_name}` (self-healing disabled)",
                )
            if prior_self_heals >= self.config.max_self_heal_retries:
                return _SingleToolOutcome(
                    message=msg,
                    fatal_error=f"Exceeded max_self_heal_retries={self.config.max_self_heal_retries} on `{tool_name}`",
                )
            return _SingleToolOutcome(message=msg, self_heal_error=True)
        except ToolSecurityBlockedError as exc:
            sec_payload = {
                "ok": False,
                "error_type": "security_blocked",
                "reason_code": exc.reason_code,
                "tool": tool_name,
                "error": str(exc),
            }
            return _SingleToolOutcome(
                message=Message(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tool_name,
                    content=json.dumps(sec_payload, ensure_ascii=False),
                ),
                security_blocked=True,
            )
        except Exception as exc:
            exec_payload = {
                "ok": False,
                "error_type": "execution_error",
                "tool": tool_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _SingleToolOutcome(
                message=Message(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tool_name,
                    content=json.dumps(exec_payload, ensure_ascii=False),
                )
            )
