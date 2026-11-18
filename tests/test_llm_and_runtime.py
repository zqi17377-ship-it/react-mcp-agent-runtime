"""Unit tests for OpenAI HTTP client (backoff + SSE stream) and Self-Healing ReActRuntime."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from react_mcp_runtime.llm_client import (
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from react_mcp_runtime.runtime import ReActRuntime
from react_mcp_runtime.schema import Message, RuntimeConfig, ToolCall, ToolCallFunction
from react_mcp_runtime.tools import ToolRegistry, tool


async def test_openai_client_exponential_backoff_on_429_and_503() -> None:
    attempts = 0
    recorded_sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.15"}, json={"error": "rate_limit"})
        if attempts == 2:
            return httpx.Response(503, json={"error": "upstream_overloaded"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Service recovered.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAICompatibleLLMClient(
            api_key="sk-test-key",
            base_url="https://mock.openai.local/v1",
            model="deepseek-chat",
            base_delay=0.2,
            http_client=http_client,
            sleep_fn=fake_sleep,
        )
        resp = await llm.complete([Message(role="user", content="ping")])

    assert attempts == 3
    assert resp.retries == 2
    assert len(recorded_sleeps) == 2
    assert recorded_sleeps[0] == 0.15  # Honored Retry-After header
    assert recorded_sleeps[1] >= 0.4   # Exponential 2^1 * base_delay
    assert resp.message.content == "Service recovered."
    assert resp.usage.total_tokens == 50


async def test_openai_client_async_sse_streaming_parser() -> None:
    sse_lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Checking "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"metrics..."}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_sse_1","type":"function","function":{"name":"query_","arguments":"{\\"svc\\":"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"metrics","arguments":"\\"checkout\\"}"}}]}}]}\n\n',
        'data: {"choices":[],"usage":{"prompt_tokens":64,"completion_tokens":21,"total_tokens":85}}\n\n',
        "data: [DONE]\n\n",
    ]

    class FakeByteStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in sse_lines:
                yield chunk.encode("utf-8")

    def stream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=FakeByteStream(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(stream_handler)) as http_client:
        llm = OpenAICompatibleLLMClient(
            api_key="sk-test-key",
            base_url="https://mock.openai.local/v1",
            model="gpt-4o-mini",
            http_client=http_client,
        )
        resp = await llm.complete([Message(role="user", content="stream test")], stream=True)

    assert resp.message.content == "Checking metrics..."
    assert resp.message.tool_calls is not None and len(resp.message.tool_calls) == 1
    tc = resp.message.tool_calls[0]
    assert tc.id == "call_sse_1"
    assert tc.function.name == "query_metrics"
    assert json.loads(tc.function.arguments) == {"svc": "checkout"}
    assert resp.usage.prompt_tokens == 64
    assert resp.usage.completion_tokens == 21


async def test_react_runtime_self_heals_malformed_json_and_validation_error() -> None:
    @tool(description="Add two bounded integers.")
    def add_ints(a: int, b: int) -> dict[str, int]:
        return {"sum": a + b}

    registry = ToolRegistry()
    registry.register(add_ints)

    scripted = [
        # Step 1: Truncated / broken JSON syntax
        Message(
            role="assistant",
            content="Calling add_ints with broken JSON.",
            tool_calls=[
                ToolCall(
                    id="c_bad_json",
                    function=ToolCallFunction(name="add_ints", arguments='{"a": 10, "b": '),
                )
            ],
        ),
        # Step 2: Valid JSON syntax, but wrong parameter type (`b` is non-numeric string)
        Message(
            role="assistant",
            content="Retrying with wrong type.",
            tool_calls=[
                ToolCall(
                    id="c_bad_type",
                    function=ToolCallFunction(
                        name="add_ints", arguments=json.dumps({"a": 10, "b": "not_an_int"})
                    ),
                )
            ],
        ),
        # Step 3: Properly repaired tool call after inspecting validation_error feedback
        Message(
            role="assistant",
            content="Repaired types.",
            tool_calls=[
                ToolCall(
                    id="c_good",
                    function=ToolCallFunction(name="add_ints", arguments=json.dumps({"a": 10, "b": 32})),
                )
            ],
        ),
        # Step 4: Final answer
        Message(role="assistant", content="The verified sum is 42."),
    ]

    runtime = ReActRuntime(
        llm_client=MockLLMClient(scripted_responses=scripted),
        tool_registry=registry,
        config=RuntimeConfig(max_steps=6, max_self_heal_retries=3, enable_self_healing=True),
    )
    result = await runtime.run("Compute 10 + 32")

    assert result.status == "completed"
    assert result.self_heal_count == 2
    assert result.steps == 4
    assert "42" in result.final_answer


async def test_react_runtime_self_heal_circuit_breaker_aborts_cleanly() -> None:
    @tool()
    def echo_num(n: int) -> int:
        return n

    registry = ToolRegistry()
    registry.register(echo_num)

    scripted = [
        Message(
            role="assistant",
            tool_calls=[ToolCall(id=f"bad_{i}", function=ToolCallFunction(name="echo_num", arguments="{broken"))]
        )
        for i in range(5)
    ]
    runtime = ReActRuntime(
        llm_client=MockLLMClient(scripted_responses=scripted),
        tool_registry=registry,
        config=RuntimeConfig(max_steps=8, max_self_heal_retries=2, enable_self_healing=True),
    )
    result = await runtime.run("Trigger circuit breaker")

    assert result.status == "aborted"
    assert result.self_heal_count == 2
    assert result.steps == 3  # 2 self-healed attempts + 3rd trips circuit breaker cleanly


async def test_react_runtime_parallel_tool_execution() -> None:
    @tool()
    async def io_probe(probe_id: int) -> dict[str, int]:
        await asyncio.sleep(0.08)
        return {"probe_id": probe_id}

    registry = ToolRegistry()
    registry.register(io_probe)

    scripted = [
        Message(
            role="assistant",
            content="Running 3 probes concurrently.",
            tool_calls=[
                ToolCall(id="p1", function=ToolCallFunction(name="io_probe", arguments='{"probe_id": 1}')),
                ToolCall(id="p2", function=ToolCallFunction(name="io_probe", arguments='{"probe_id": 2}')),
                ToolCall(id="p3", function=ToolCallFunction(name="io_probe", arguments='{"probe_id": 3}')),
            ],
        ),
        Message(role="assistant", content="All 3 probes finished."),
    ]
    runtime = ReActRuntime(
        llm_client=MockLLMClient(scripted_responses=scripted, simulated_latency_ms=0.1),
        tool_registry=registry,
        config=RuntimeConfig(parallel_tool_calls=True),
    )

    t0 = time.perf_counter()
    result = await runtime.run("Run parallel probes")
    elapsed = time.perf_counter() - t0

    assert result.status == "completed"
    # Sequential would take >= 0.24s; parallel finishes well under 0.21s
    assert elapsed < 0.22


async def test_naive_mode_aborts_on_malformed_json_and_context_overflow() -> None:
    @tool()
    def noop(x: str) -> str:
        return x

    registry = ToolRegistry()
    registry.register(noop)

    # 1. Self-healing disabled -> immediate abort on malformed JSON
    bad_json_runtime = ReActRuntime(
        llm_client=MockLLMClient(
            scripted_responses=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="b1", function=ToolCallFunction(name="noop", arguments="{bad"))],
                )
            ]
        ),
        tool_registry=registry,
        config=RuntimeConfig(enable_self_healing=False),
    )
    res_bad = await bad_json_runtime.run("test")
    assert res_bad.status == "aborted"
    assert res_bad.self_heal_count == 0

    # 2. Compression disabled -> aborts when prompt exceeds hard_context_limit
    overflow_runtime = ReActRuntime(
        llm_client=MockLLMClient(hard_context_limit=60),
        tool_registry=registry,
        config=RuntimeConfig(enable_compression=False),
    )
    res_overflow = await overflow_runtime.run("X" * 400)
    assert res_overflow.status == "aborted"
    assert "context_length_exceeded" in res_overflow.final_answer
