"""Dual-Mode Async LLM Client: OpenAI-Compatible HTTP (SSE + Retry) & Scenario-Aware Mock."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from react_mcp_runtime.context import SUMMARY_MARKER, estimate_message_tokens, estimate_tokens
from react_mcp_runtime.schema import LLMResponse, LLMUsage, Message, ToolCall, ToolCallFunction

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ContextLengthExceededError(RuntimeError):
    """Raised when prompt tokens exceed the model's context window budget."""


class BaseLLMClient(ABC):
    """Unified interface for async LLM chat completions with tool calling."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> LLMResponse:
        """Generate the next assistant turn given conversation history and tool schemas."""


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Zero-framework OpenAI/DeepSeek/Qwen HTTP client with SSE streaming & exponential backoff."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 45.0,
        max_retries: int = 3,
        base_delay: float = 0.3,
        max_delay: float = 4.0,
        temperature: float = 0.0,
        http_client: httpx.AsyncClient | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        raw_base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.base_url = raw_base.rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.temperature = temperature
        self._external_client = http_client
        self._sleep_fn = sleep_fn

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai_dict() for m in messages],
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _compute_backoff(self, attempt: int, retry_after_header: str | None = None) -> float:
        if retry_after_header:
            try:
                parsed = float(retry_after_header)
                if 0 < parsed <= self.max_delay:
                    return parsed
            except ValueError:
                pass
        exp = min(self.max_delay, self.base_delay * (2**attempt))
        jitter = exp * random.uniform(0.0, 0.2)
        return round(exp + jitter, 4)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> LLMResponse:
        payload = self._build_payload(messages, tools, stream=stream)
        endpoint = f"{self.base_url}/chat/completions"
        retries = 0
        start_ts = time.perf_counter()

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    if stream:
                        msg, usage = await self._send_streaming(client, endpoint, payload, messages)
                    else:
                        msg, usage = await self._send_non_streaming(client, endpoint, payload, messages)
                    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                    return LLMResponse(
                        message=msg,
                        usage=usage,
                        model=self.model,
                        latency_ms=elapsed_ms,
                        retries=retries,
                    )
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    body_text = exc.response.text or ""
                    if status == 400 and "context_length" in body_text.lower():
                        raise ContextLengthExceededError(body_text) from exc
                    if status in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        retries += 1
                        delay = self._compute_backoff(
                            attempt, exc.response.headers.get("Retry-After")
                        )
                        await self._sleep_fn(delay)
                        continue
                    raise
                except (httpx.TransportError, httpx.TimeoutException):
                    if attempt < self.max_retries:
                        retries += 1
                        await self._sleep_fn(self._compute_backoff(attempt))
                        continue
                    raise
        finally:
            if owns_client:
                await client.aclose()

        raise RuntimeError("Unreachable retry loop exit")

    async def _send_non_streaming(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, Any],
        messages: list[Message],
    ) -> tuple[Message, LLMUsage]:
        resp = await client.post(endpoint, headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        raw_tcs = choice.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc["id"],
                type="function",
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"].get("arguments") or "{}",
                ),
            )
            for tc in raw_tcs
        ] or None

        assistant_msg = Message(
            role="assistant",
            content=choice.get("content"),
            tool_calls=tool_calls,
        )
        raw_usage = data.get("usage") or {}
        p_tok = int(raw_usage.get("prompt_tokens") or estimate_tokens(messages))
        c_tok = int(raw_usage.get("completion_tokens") or estimate_message_tokens(assistant_msg))
        usage = LLMUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=int(raw_usage.get("total_tokens") or (p_tok + c_tok)),
        )
        return assistant_msg, usage

    async def _send_streaming(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, Any],
        messages: list[Message],
    ) -> tuple[Message, LLMUsage]:
        content_chunks: list[str] = []
        tc_accum: dict[int, dict[str, str]] = {}
        stream_usage: dict[str, int] | None = None

        async with client.stream("POST", endpoint, headers=self._headers(), json=payload) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                resp.raise_for_status()

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                frame = json.loads(data_str)
                if frame.get("usage"):
                    stream_usage = frame["usage"]
                choices = frame.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content_chunks.append(delta["content"])
                for tc_delta in delta.get("tool_calls") or []:
                    idx = int(tc_delta.get("index", 0))
                    slot = tc_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc_delta.get("id"):
                        slot["id"] = tc_delta["id"]
                    fn_part = tc_delta.get("function") or {}
                    if fn_part.get("name"):
                        slot["name"] += fn_part["name"]
                    if fn_part.get("arguments"):
                        slot["arguments"] += fn_part["arguments"]

        tool_calls: list[ToolCall] | None = None
        if tc_accum:
            tool_calls = [
                ToolCall(
                    id=tc_accum[idx]["id"] or f"call_stream_{idx}",
                    type="function",
                    function=ToolCallFunction(
                        name=tc_accum[idx]["name"],
                        arguments=tc_accum[idx]["arguments"] or "{}",
                    ),
                )
                for idx in sorted(tc_accum)
            ]

        full_content = "".join(content_chunks) if content_chunks else None
        assistant_msg = Message(role="assistant", content=full_content, tool_calls=tool_calls)
        if stream_usage:
            p_tok = int(stream_usage.get("prompt_tokens", 0))
            c_tok = int(stream_usage.get("completion_tokens", 0))
            usage = LLMUsage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=int(stream_usage.get("total_tokens") or (p_tok + c_tok)),
            )
        else:
            p_tok = estimate_tokens(messages)
            c_tok = estimate_message_tokens(assistant_msg)
            usage = LLMUsage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok)
        return assistant_msg, usage


class MockLLMClient(BaseLLMClient):
    """Deterministic, schema- & history-inspecting offline LLM for ReAct + MCP verification."""

    def __init__(
        self,
        *,
        model: str = "mock-react-v1",
        scripted_responses: list[Message] | None = None,
        hard_context_limit: int | None = None,
        simulated_latency_ms: float = 4.5,
    ) -> None:
        self.model = model
        self._scripted: deque[Message] = deque(scripted_responses or [])
        self.hard_context_limit = hard_context_limit
        self.simulated_latency_ms = simulated_latency_ms
        self._call_counter = 0

    def _next_call_id(self, suffix: str = "a") -> str:
        self._call_counter += 1
        return f"call_{self._call_counter:03d}_{suffix}"

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> LLMResponse:
        start_ts = time.perf_counter()
        p_tok = estimate_tokens(messages)
        if self.hard_context_limit is not None and p_tok > self.hard_context_limit:
            raise ContextLengthExceededError(
                f"Prompt tokens ({p_tok}) exceeded model hard_context_limit ({self.hard_context_limit})."
            )

        if self._scripted:
            reply = self._scripted.popleft()
        else:
            reply = self._decide_turn(messages, tools or [])

        c_tok = estimate_message_tokens(reply)
        elapsed_ms = max(self.simulated_latency_ms, (time.perf_counter() - start_ts) * 1000.0)
        return LLMResponse(
            message=reply,
            usage=LLMUsage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok),
            model=self.model,
            latency_ms=elapsed_ms,
            retries=0,
        )

    def _extract_user_goal(self, messages: list[Message]) -> str:
        for m in messages:
            if m.role == "user" and m.content and not m.content.startswith(SUMMARY_MARKER):
                return m.content
        return ""

    def _collect_state(self, messages: list[Message]) -> dict[str, Any]:
        """Inspect live tool messages AND compressed state summaries to reconstruct agent knowledge."""
        executed_tools: list[str] = []
        tool_outputs: dict[str, list[Any]] = {}
        error_turns: list[dict[str, Any]] = []
        security_blocks: list[dict[str, Any]] = []
        summary_text = ""

        tc_lookup: dict[str, ToolCall] = {}
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    tc_lookup[tc.id] = tc
            elif m.role == "user" and m.content and m.content.startswith(SUMMARY_MARKER):
                summary_text = m.content
                for t_name in (
                    "analyze_service_logs",
                    "sqlite_readonly_query",
                    "git_commit_history",
                    "git_show_diff",
                    "apply_incident_remediation",
                ):
                    if f"- {t_name} ->" in summary_text and t_name not in executed_tools:
                        executed_tools.append(t_name)
            elif m.role == "tool":
                origin_tc = tc_lookup.get(m.tool_call_id or "")
                t_name = (origin_tc.function.name if origin_tc else m.name) or "unknown"
                parsed: Any = m.content or ""
                try:
                    parsed = json.loads(m.content or "{}")
                except Exception:
                    pass
                if isinstance(parsed, dict) and parsed.get("ok") is False:
                    err_type = parsed.get("error_type", "")
                    if err_type in ("json_decode_error", "validation_error"):
                        error_turns.append({"tool": t_name, "detail": parsed, "tc": origin_tc})
                    elif err_type == "security_blocked":
                        security_blocks.append({"tool": t_name, "detail": parsed})
                        executed_tools.append(f"blocked:{t_name}")
                    else:
                        executed_tools.append(t_name)
                        tool_outputs.setdefault(t_name, []).append(parsed)
                else:
                    executed_tools.append(t_name)
                    tool_outputs.setdefault(t_name, []).append(parsed)

        return {
            "executed_tools": executed_tools,
            "tool_outputs": tool_outputs,
            "error_turns": error_turns,
            "security_blocks": security_blocks,
            "summary_text": summary_text,
        }

    def _repair_broken_call(self, err_record: dict[str, Any], error_count_so_far: int, goal: str) -> Message:
        """Read structured self-healing feedback and emit a valid repaired tool call."""
        t_name = err_record["tool"]
        if "[FAULT:PERSISTENT_BAD_SCHEMA_4X]" in goal and error_count_so_far < 4:
            # Intentional persistent hallucination scenario to test max_self_heal_retries circuit breaker
            return Message(
                role="assistant",
                content="Retrying tool call (persistent schema drift stress test).",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("bad"),
                        function=ToolCallFunction(
                            name=t_name,
                            arguments=json.dumps({"limit": "still_not_an_integer", "bogus": True}),
                        ),
                    )
                ],
            )

        if "[FAULT:DOUBLE_CORRUPT]" in goal and error_count_so_far == 1:
            # Step 1 had malformed JSON; Step 2 injects a schema type error before final repair on Step 3
            return Message(
                role="assistant",
                content="Fixed JSON syntax; calling git_commit_history next.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("schema_err"),
                        function=ToolCallFunction(
                            name="git_commit_history",
                            arguments=json.dumps({"limit": "five"}),
                        ),
                    )
                ],
            )

        repaired_args_map: dict[str, dict[str, Any]] = {
            "analyze_service_logs": {"service": "checkout-api", "min_level": "ERROR", "top_k": 5},
            "git_commit_history": {"limit": 6},
            "git_show_diff": {"revision": "v2.14.3"},
            "sqlite_readonly_query": {
                "sql": "SELECT service, version, p99_ms, error_rate FROM latency_rollups ORDER BY p99_ms DESC LIMIT 5",
                "max_rows": 10,
            },
        }
        clean_args = repaired_args_map.get(t_name, {})
        return Message(
            role="assistant",
            content=f"Self-healing `{t_name}` arguments after inspecting validation feedback.",
            tool_calls=[
                ToolCall(
                    id=self._next_call_id("repaired"),
                    function=ToolCallFunction(
                        name=t_name,
                        arguments=json.dumps(clean_args),
                    ),
                )
            ],
        )

    def _decide_turn(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        goal = self._extract_user_goal(messages)
        available = {
            t["function"]["name"]: t["function"]
            for t in tools
            if t.get("type") == "function" and "function" in t
        }
        state = self._collect_state(messages)
        executed = state["executed_tools"]
        outputs = state["tool_outputs"]
        error_turns = state["error_turns"]
        security_blocks = state["security_blocks"]

        # 1. If the immediately preceding turn was a malformed JSON / validation error, repair it!
        if messages and messages[-1].role == "tool":
            try:
                last_payload = json.loads(messages[-1].content or "{}")
            except Exception:
                last_payload = {}
            if isinstance(last_payload, dict) and last_payload.get("error_type") in (
                "json_decode_error",
                "validation_error",
            ):
                return self._repair_broken_call(error_turns[-1], len(error_turns), goal)

        # 2. Fault injection on initial turn (before any tool has succeeded or failed)
        if not executed and not error_turns and not security_blocks:
            fault_call = self._maybe_inject_initial_fault(goal, available)
            if fault_call is not None:
                return fault_call

        # 3. Adversarial SQL injection test scenarios
        sql_attack = self._maybe_inject_sql_attack(goal, available, executed, security_blocks)
        if sql_attack is not None:
            return sql_attack

        # 4. Premature / sneaky write governance test scenarios
        hitl_attack = self._maybe_inject_hitl_scenario(goal, available, executed, security_blocks)
        if hitl_attack is not None:
            return hitl_attack

        # 5. Standard & Deep Forensic MCP Incident Triage Workflow
        has_mcp_suite = {"analyze_service_logs", "sqlite_readonly_query", "git_commit_history"}.issubset(
            available.keys()
        )
        if has_mcp_suite:
            return self._drive_incident_triage(goal, available, state)

        # 6. Generic fallback for arbitrary unit-test custom tools
        for t_name, fn_spec in available.items():
            if t_name not in executed and f"blocked:{t_name}" not in executed:
                props = fn_spec.get("parameters", {}).get("properties", {})
                sample_args: dict[str, Any] = {}
                for k, p in props.items():
                    if "default" in p:
                        sample_args[k] = p["default"]
                    elif p.get("type") == "integer":
                        sample_args[k] = 3
                    elif p.get("type") == "number":
                        sample_args[k] = 1.0
                    elif p.get("type") == "boolean":
                        sample_args[k] = True
                    else:
                        sample_args[k] = "test"
                return Message(
                    role="assistant",
                    content=f"Invoking registered tool `{t_name}`.",
                    tool_calls=[
                        ToolCall(
                            id=self._next_call_id("gen"),
                            function=ToolCallFunction(name=t_name, arguments=json.dumps(sample_args)),
                        )
                    ],
                )

        return Message(
            role="assistant",
            content=f"Completed task using tools: {', '.join(executed) or 'none'}.",
        )

    def _maybe_inject_initial_fault(
        self, goal: str, available: dict[str, Any]
    ) -> Message | None:
        if "analyze_service_logs" not in available and "git_commit_history" not in available:
            return None
        if "[FAULT:TRUNCATED_JSON]" in goal or "[FAULT:DOUBLE_CORRUPT]" in goal:
            return Message(
                role="assistant",
                content="Checking checkout-api error logs.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("trunc"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments='{"service": "checkout-api", "min_level": "ERROR"',
                        ),
                    )
                ],
            )
        if "[FAULT:MALFORMED_JSON]" in goal:
            return Message(
                role="assistant",
                content="Inspecting service logs for error signatures.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("mal"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments="{service: 'checkout-api', min_level: 'ERROR',}",
                        ),
                    )
                ],
            )
        if "[FAULT:WRONG_TYPE]" in goal or "[FAULT:PERSISTENT_BAD_SCHEMA_4X]" in goal:
            return Message(
                role="assistant",
                content="Fetching recent git commit history.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("type"),
                        function=ToolCallFunction(
                            name="git_commit_history",
                            arguments=json.dumps({"limit": "six_commits"}),
                        ),
                    )
                ],
            )
        if "[FAULT:EXTRA_ARG]" in goal:
            return Message(
                role="assistant",
                content="Inspecting commit diff with formatting option.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("extra"),
                        function=ToolCallFunction(
                            name="git_show_diff",
                            arguments=json.dumps({"revision": "v2.14.3", "render_html": True}),
                        ),
                    )
                ],
            )
        if "[FAULT:BAD_ENUM]" in goal:
            return Message(
                role="assistant",
                content="Filtering logs by severity.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("enum"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments=json.dumps({"service": "checkout-api", "min_level": "FATAL_SEV1"}),
                        ),
                    )
                ],
            )
        return None

    def _maybe_inject_sql_attack(
        self,
        goal: str,
        available: dict[str, Any],
        executed: list[str],
        security_blocks: list[dict[str, Any]],
    ) -> Message | None:
        if "sqlite_readonly_query" not in available:
            return None
        attack_sql: str | None = None
        if "[FAULT:SQL_DROP]" in goal:
            attack_sql = "SELECT service, version FROM deployments; DROP TABLE error_events;"
        elif "[FAULT:SQL_COMMENT_DELETE]" in goal:
            attack_sql = "/* cleanup noisy metrics */ DELETE FROM latency_rollups WHERE error_rate > 0.10"
        elif "[FAULT:SQL_UPDATE]" in goal:
            attack_sql = "UPDATE deployments SET status = 'healthy' WHERE version = 'v2.14.3'"
        elif "[FAULT:SQL_ATTACH]" in goal:
            attack_sql = "ATTACH DATABASE ':memory:' AS exfil_db"

        if attack_sql is None:
            return None
        if not executed and not security_blocks:
            return Message(
                role="assistant",
                content="Executing requested SQL statement against service metrics.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("sql_atk"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps({"sql": attack_sql, "max_rows": 20}),
                        ),
                    )
                ],
            )
        # After SQL write attempt was blocked (or executed in unguarded baseline), perform safe read fallback
        if "sqlite_readonly_query" not in executed:
            return Message(
                role="assistant",
                content="Destructive SQL was blocked by read-only guard; running safe SELECT diagnostics instead.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("sql_safe"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps(
                                {
                                    "sql": "SELECT service, version, p99_ms, error_rate FROM latency_rollups ORDER BY p99_ms DESC LIMIT 4",
                                    "max_rows": 10,
                                }
                            ),
                        ),
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "Security Guard Report: Destructive SQL mutation was intercepted and refused by `SQLReadOnlyGuard`. "
                "Read-only diagnostics confirm `checkout-api` `v2.14.3` experienced `p99_ms=1840.0` (`error_rate=0.187`)."
            ),
        )

    def _maybe_inject_hitl_scenario(
        self,
        goal: str,
        available: dict[str, Any],
        executed: list[str],
        security_blocks: list[dict[str, Any]],
    ) -> Message | None:
        if "apply_incident_remediation" not in available:
            return None
        if "[ACTION:PREMATURE_REMEDIATE]" in goal:
            if not executed and not security_blocks:
                return Message(
                    role="assistant",
                    content="Attempting immediate rollback tag before root-cause verification.",
                    tool_calls=[
                        ToolCall(
                            id=self._next_call_id("premature"),
                            function=ToolCallFunction(
                                name="apply_incident_remediation",
                                arguments=json.dumps(
                                    {
                                        "action": "tag_rollback_target",
                                        "target_commit": "HEAD",
                                        "incident_id": "INC-UNVERIFIED",
                                        "reason": "Unverified emergency rollback attempt",
                                    }
                                ),
                            ),
                        )
                    ],
                )
            return Message(
                role="assistant",
                content=(
                    "Write Governance Report: Unverified `apply_incident_remediation` call was blocked by HITL policy "
                    f"({len(security_blocks)} security block recorded). No unauthorized tag was applied."
                ),
            )

        if "[FAULT:SNEAKY_PARALLEL_WRITE]" in goal:
            if not executed and not security_blocks:
                return Message(
                    role="assistant",
                    content="Running log analysis alongside remediation action in parallel.",
                    tool_calls=[
                        ToolCall(
                            id=self._next_call_id("par_read"),
                            function=ToolCallFunction(
                                name="analyze_service_logs",
                                arguments=json.dumps({"service": "checkout-api", "min_level": "ERROR"}),
                            ),
                        ),
                        ToolCall(
                            id=self._next_call_id("par_write"),
                            function=ToolCallFunction(
                                name="apply_incident_remediation",
                                arguments=json.dumps(
                                    {
                                        "action": "quarantine_release",
                                        "target_commit": "v2.14.3",
                                        "incident_id": "INC-SNEAKY",
                                        "reason": "Unapproved parallel write injection",
                                    }
                                ),
                            ),
                        ),
                    ],
                )
            return Message(
                role="assistant",
                content=(
                    "Parallel Batch Governance Report: Read-only `analyze_service_logs` succeeded (`PoolExhaustedTimeoutError` on `checkout-api`), "
                    "while unauthorized `apply_incident_remediation` in the same batch was intercepted and blocked by HITL security policy."
                ),
            )
        return None

    def _drive_incident_triage(
        self,
        goal: str,
        available: dict[str, Any],
        state: dict[str, Any],
    ) -> Message:
        executed = state["executed_tools"]
        outputs = state["tool_outputs"]
        summary_text = state["summary_text"]
        deep_mode = "[MODE:DEEP_FORENSIC]" in goal
        wants_remediation = "[ACTION:REMEDIATE]" in goal

        # Step 1: Parallel read of service error logs + latency rollup SQLite table
        has_logs = "analyze_service_logs" in executed
        sql_calls_done = len(outputs.get("sqlite_readonly_query", [])) + summary_text.count(
            "- sqlite_readonly_query ->"
        )
        if not has_logs and sql_calls_done == 0:
            return Message(
                role="assistant",
                content=(
                    "Correlating structured error logs and SQLite latency/error rollups in parallel "
                    "to isolate the anomalous service and deployment window."
                ),
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("logs"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments=json.dumps(
                                {"service": "checkout-api", "min_level": "ERROR", "top_k": 5}
                            ),
                        ),
                    ),
                    ToolCall(
                        id=self._next_call_id("lat"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps(
                                {
                                    "sql": (
                                        "SELECT d.deploy_id, d.service, d.version, d.git_commit, "
                                        "l.p50_ms, l.p99_ms, l.error_rate "
                                        "FROM latency_rollups l "
                                        "JOIN deployments d ON l.service = d.service AND l.version = d.version "
                                        "ORDER BY l.p99_ms DESC LIMIT 6"
                                    ),
                                    "max_rows": 10,
                                }
                            ),
                        ),
                    ),
                ],
            )

        if not has_logs:
            return Message(
                role="assistant",
                content="Analyzing checkout-api JSONL error logs to identify top exception signatures.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("logs_only"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments=json.dumps(
                                {"service": "checkout-api", "min_level": "ERROR", "top_k": 5}
                            ),
                        ),
                    )
                ],
            )

        if sql_calls_done == 0:
            return Message(
                role="assistant",
                content="Querying SQLite deployments and latency_rollups to pinpoint regression release.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("sql_join"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps(
                                {
                                    "sql": (
                                        "SELECT d.deploy_id, d.service, d.version, d.git_commit, "
                                        "l.p99_ms, l.error_rate "
                                        "FROM latency_rollups l "
                                        "JOIN deployments d ON l.service = d.service AND l.version = d.version "
                                        "ORDER BY l.p99_ms DESC LIMIT 6"
                                    ),
                                    "max_rows": 10,
                                }
                            ),
                        ),
                    )
                ],
            )

        # Step 2: Inspect Git commit history (+ error_events table in deep mode)
        if "git_commit_history" not in executed:
            calls = [
                ToolCall(
                    id=self._next_call_id("gitlog"),
                    function=ToolCallFunction(
                        name="git_commit_history",
                        arguments=json.dumps({"limit": 6}),
                    ),
                )
            ]
            if deep_mode:
                calls.append(
                    ToolCall(
                        id=self._next_call_id("err_tbl"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps(
                                {
                                    "sql": "SELECT service, version, error_code, exception_type, count FROM error_events ORDER BY count DESC LIMIT 10",
                                    "max_rows": 10,
                                }
                            ),
                        ),
                    )
                )
            return Message(
                role="assistant",
                content="Retrieving Git commit timeline to map faulty release v2.14.3 and last-known-good v2.14.2.",
                tool_calls=calls,
            )

        # Step 3: Inspect unified diff of faulty commit / tag v2.14.3
        diff_calls_done = len(outputs.get("git_show_diff", [])) + summary_text.count(
            "- git_show_diff ->"
        )
        if diff_calls_done == 0 and "git_show_diff" in available:
            suspect_rev = self._find_suspect_revision(outputs, summary_text) or "v2.14.3"
            return Message(
                role="assistant",
                content=f"Inspecting patch diff for suspect revision `{suspect_rev}`.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("diff1"),
                        function=ToolCallFunction(
                            name="git_show_diff",
                            arguments=json.dumps({"revision": suspect_rev}),
                        ),
                    )
                ],
            )

        # Deep Forensic Mode extra verification turns (triggers sliding-window context compression cleanly)
        if deep_mode and diff_calls_done < 2 and "git_show_diff" in available:
            return Message(
                role="assistant",
                content="Cross-checking baseline release `v2.14.2` diff and healthy sibling service logs (`catalog-api`, `auth-api`).",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("diff_base"),
                        function=ToolCallFunction(
                            name="git_show_diff",
                            arguments=json.dumps({"revision": "v2.14.2"}),
                        ),
                    ),
                    ToolCall(
                        id=self._next_call_id("logs_all"),
                        function=ToolCallFunction(
                            name="analyze_service_logs",
                            arguments=json.dumps({"min_level": "WARN", "top_k": 5}),
                        ),
                    ),
                ],
            )

        if deep_mode and sql_calls_done < 3:
            return Message(
                role="assistant",
                content="Verifying complete deployment chronology across all services in SQLite.",
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("deploy_all"),
                        function=ToolCallFunction(
                            name="sqlite_readonly_query",
                            arguments=json.dumps(
                                {
                                    "sql": "SELECT deploy_id, service, version, git_commit, deployed_at, status FROM deployments ORDER BY deployed_at DESC",
                                    "max_rows": 15,
                                }
                            ),
                        ),
                    )
                ],
            )

        # Optional Step 4: Remediation write tool (guarded by HITL policy on MCP client)
        if (
            wants_remediation
            and "apply_incident_remediation" in available
            and "apply_incident_remediation" not in executed
            and "blocked:apply_incident_remediation" not in executed
        ):
            rollback_target = self._find_rollback_commit(outputs, summary_text) or "v2.14.2"
            return Message(
                role="assistant",
                content=(
                    f"Root cause confirmed in `v2.14.3` (`pool_max_size` dropped 32->3). "
                    f"Requesting HITL approval to tag rollback target `{rollback_target}`."
                ),
                tool_calls=[
                    ToolCall(
                        id=self._next_call_id("remediate"),
                        function=ToolCallFunction(
                            name="apply_incident_remediation",
                            arguments=json.dumps(
                                {
                                    "action": "tag_rollback_target",
                                    "target_commit": rollback_target,
                                    "incident_id": "INC-2026-0315",
                                    "reason": "Rollback checkout-api from v2.14.3 (db pool starvation) to healthy v2.14.2",
                                }
                            ),
                        ),
                    )
                ],
            )

        # Final Step: Synthesize evidence-grounded report from actual tool outputs & summary
        return Message(
            role="assistant",
            content=self._synthesize_triage_report(outputs, state["security_blocks"], summary_text),
        )

    def _find_suspect_revision(self, outputs: dict[str, list[Any]], summary_text: str) -> str | None:
        for res in outputs.get("sqlite_readonly_query", []):
            if isinstance(res, dict):
                for row in res.get("rows", []):
                    if row.get("version") == "v2.14.3" and row.get("git_commit"):
                        return str(row["git_commit"])
        for res in outputs.get("git_commit_history", []):
            if isinstance(res, dict):
                for c in res.get("commits", []):
                    if "tune db pool" in str(c.get("subject", "")).lower():
                        return str(c.get("short_hash") or c.get("hash"))
        match = re.search(r"([0-9a-f]{7,40}):chore\(config\): tune db pool", summary_text)
        if match:
            return match.group(1)
        return None

    def _find_rollback_commit(self, outputs: dict[str, list[Any]], summary_text: str) -> str | None:
        for res in outputs.get("sqlite_readonly_query", []):
            if isinstance(res, dict):
                for row in res.get("rows", []):
                    if row.get("version") == "v2.14.2" and row.get("git_commit"):
                        return str(row["git_commit"])[:8]
        for res in outputs.get("git_commit_history", []):
            if isinstance(res, dict):
                for c in res.get("commits", []):
                    if "clock-skew" in str(c.get("subject", "")).lower():
                        return str(c.get("short_hash") or c.get("hash"))[:8]
        match = re.search(r"([0-9a-f]{7,40}):refactor\(auth\)", summary_text)
        if match:
            return match.group(1)[:8]
        return "v2.14.2"

    def _synthesize_triage_report(
        self,
        outputs: dict[str, list[Any]],
        security_blocks: list[dict[str, Any]],
        summary_text: str,
    ) -> str:
        top_exc = "PoolExhaustedTimeoutError"
        err_count = 0
        for log_res in outputs.get("analyze_service_logs", []):
            if isinstance(log_res, dict) and log_res.get("top_exceptions"):
                first_ex = log_res["top_exceptions"][0]
                top_exc = str(first_ex.get("exception_type", top_exc))
                err_count = int(first_ex.get("count", 0))
                break

        peak_p99 = 1840.0
        peak_err_rate = 0.187
        suspect_commit = self._find_suspect_revision(outputs, summary_text) or "v2.14.3"
        rollback_commit = self._find_rollback_commit(outputs, summary_text) or "v2.14.2"

        for sql_res in outputs.get("sqlite_readonly_query", []):
            if isinstance(sql_res, dict):
                for row in sql_res.get("rows", []):
                    if row.get("service") == "checkout-api" and row.get("p99_ms"):
                        val = float(row["p99_ms"])
                        if val > peak_p99 or peak_p99 == 1840.0:
                            peak_p99 = val
                        if row.get("error_rate") is not None:
                            peak_err_rate = float(row["error_rate"])
                        break

        remediation_note = "Read-only investigation completed (no write actions requested)."
        if outputs.get("apply_incident_remediation"):
            rem = outputs["apply_incident_remediation"][-1]
            tag_name = rem.get("tag_created", "rollback/INC-2026-0315") if isinstance(rem, dict) else "rollback/INC-2026-0315"
            remediation_note = f"Approved via HITL; created rollback tag `{tag_name}` -> `{rollback_commit}`."
        elif security_blocks:
            remediation_note = (
                f"Write remediation intercepted and blocked by HITL security policy ({security_blocks[-1]['detail'].get('reason_code', 'denied')})."
            )

        return (
            "### Incident Root-Cause Analysis (`checkout-api`)\n"
            f"- **Impacted Service & Release**: `checkout-api` version `v2.14.3` (commit `{suspect_commit[:8]}`)\n"
            f"- **Telemetry Impact**: `p99_ms` spiked to `{peak_p99:.1f} ms` (`error_rate={peak_err_rate:.3f}`), while `catalog-api` and `auth-api` remained nominal.\n"
            f"- **Primary Exception Signature**: `{top_exc}` ({err_count or 'multiple'} structured log events).\n"
            "- **Code/Config Root Cause**: Commit `chore(config): tune db pool & enable sync audit flush per request` reduced `pool_max_size` from `32` to `3`, lowered `pool_timeout_ms` from `2500` to `120`, and enabled synchronous per-request audit flushing (`sync_audit_flush_on_checkout=true`).\n"
            f"- **Recommended Rollback Target**: `v2.14.2` (commit `{rollback_commit}`)\n"
            f"- **Remediation & Governance Status**: {remediation_note}"
        )


def create_llm_client(
    mode: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory selecting OpenAI HTTP client when configured, otherwise deterministic MockLLMClient."""
    resolved_mode = (mode or os.getenv("LLM_MODE", "auto")).strip().lower()
    api_key = (kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "")).strip()

    if resolved_mode == "mock" or (resolved_mode == "auto" and not api_key):
        mock_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"model", "scripted_responses", "hard_context_limit", "simulated_latency_ms"}
        }
        return MockLLMClient(**mock_kwargs)

    return OpenAICompatibleLLMClient(**kwargs)
