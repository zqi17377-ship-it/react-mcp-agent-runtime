"""Unit tests for Sliding-Window Context Compression (zero tool orphaning) & CostTracker."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from react_mcp_runtime.context import (
    SUMMARY_MARKER,
    SlidingWindowCompressor,
    estimate_tokens,
    validate_tool_call_integrity,
)
from react_mcp_runtime.cost_tracker import CostTracker, compute_cost_usd
from react_mcp_runtime.schema import LLMResponse, LLMUsage, Message, ToolCall, ToolCallFunction


def _make_parallel_turn(step: int, payload_size: int = 320) -> list[Message]:
    id_a = f"call_{step}_a"
    id_b = f"call_{step}_b"
    assistant = Message(
        role="assistant",
        content=f"Step {step}: querying logs and metrics in parallel.",
        tool_calls=[
            ToolCall(
                id=id_a,
                function=ToolCallFunction(
                    name="analyze_service_logs",
                    arguments=json.dumps({"service": "checkout-api"}),
                ),
            ),
            ToolCall(
                id=id_b,
                function=ToolCallFunction(
                    name="sqlite_readonly_query",
                    arguments=json.dumps({"sql": "SELECT * FROM latency_rollups"}),
                ),
            ),
        ],
    )
    tool_a = Message(
        role="tool",
        tool_call_id=id_a,
        name="analyze_service_logs",
        content=json.dumps(
            {
                "service": "checkout-api",
                "total_matched": 42,
                "padding": "X" * payload_size,
            }
        ),
    )
    tool_b = Message(
        role="tool",
        tool_call_id=id_b,
        name="sqlite_readonly_query",
        content=json.dumps(
            {
                "row_count": 3,
                "rows": [{"version": f"v2.14.{step}", "p99_ms": 1840.0}],
                "padding": "Y" * payload_size,
            }
        ),
    )
    return [assistant, tool_a, tool_b]


async def test_sliding_window_compression_never_orphans_tool_calls() -> None:
    history: list[Message] = [
        Message(role="system", content="System reliability instructions."),
        Message(role="user", content="Investigate checkout-api latency spike."),
    ]
    for step in range(1, 7):
        history.extend(_make_parallel_turn(step, payload_size=360))

    raw_tokens = estimate_tokens(history)
    compressor = SlidingWindowCompressor(max_context_tokens=950, keep_recent_groups=2)
    compressed, did_compress = await compressor.compress_if_needed(history)

    assert did_compress is True
    assert estimate_tokens(compressed) < raw_tokens
    # System prompt + initial user prompt stay pinned at indices 0 and 1
    assert compressed[0].role == "system"
    assert compressed[1].role == "user" and "Investigate checkout-api" in (compressed[1].content or "")
    # Index 2 holds the structured summary
    assert compressed[2].content is not None and compressed[2].content.startswith(SUMMARY_MARKER)
    # Critical invariant: ZERO orphaned tool messages or split parallel tool batches
    validate_tool_call_integrity(compressed)
    # Exactly 2 recent parallel turn groups (2 * 3 = 6 messages) remain after summary
    assert len(compressed[3:]) == 6


async def test_repeated_compressions_stay_bounded_and_merge_summary() -> None:
    compressor = SlidingWindowCompressor(max_context_tokens=800, keep_recent_groups=2)
    msgs: list[Message] = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="Original incident prompt."),
    ]
    compress_events = 0
    for step in range(1, 10):
        msgs.extend(_make_parallel_turn(step, payload_size=300))
        msgs, triggered = await compressor.compress_if_needed(msgs)
        if triggered:
            compress_events += 1
        validate_tool_call_integrity(msgs)

    assert compress_events >= 4
    summary_msgs = [m for m in msgs if m.content and m.content.startswith(SUMMARY_MARKER)]
    assert len(summary_msgs) == 1


def test_orphan_detector_catches_broken_histories() -> None:
    # Case 1: Orphaned role='tool' message without preceding assistant tool_calls
    orphan_history = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="tool", tool_call_id="call_999", name="foo", content="{}"),
    ]
    with pytest.raises(ValueError, match="Orphaned tool message"):
        validate_tool_call_integrity(orphan_history)

    # Case 2: Assistant emitted 2 tool_calls, but only 1 tool response followed
    incomplete_pair = [
        Message(role="system", content="sys"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", function=ToolCallFunction(name="a", arguments="{}")),
                ToolCall(id="c2", function=ToolCallFunction(name="b", arguments="{}")),
            ],
        ),
        Message(role="tool", tool_call_id="c1", name="a", content="{}"),
    ]
    with pytest.raises(ValueError, match="Broken assistant/tool pair"):
        validate_tool_call_integrity(incomplete_pair)


def test_cost_tracker_ledger_and_rich_panel() -> None:
    tracker = CostTracker()
    tracker.record(
        1,
        LLMResponse(
            message=Message(role="assistant", content="step 1"),
            usage=LLMUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
            model="deepseek-v3",
            latency_ms=120.5,
            retries=1,
        ),
    )
    tracker.record(
        2,
        LLMResponse(
            message=Message(role="assistant", content="step 2"),
            usage=LLMUsage(prompt_tokens=1500, completion_tokens=300, total_tokens=1800),
            model="deepseek-v3",
            latency_ms=245.0,
            retries=0,
        ),
    )

    summary = tracker.summary_dict()
    assert summary["llm_calls"] == 2
    assert summary["prompt_tokens"] == 2500
    assert summary["completion_tokens"] == 500
    assert summary["total_tokens"] == 3000
    assert summary["retry_count"] == 1
    assert summary["p95_latency_ms"] == 245.0
    expected_ds = compute_cost_usd("deepseek-v3", 2500, 500)
    assert summary["estimated_cost_usd"] == round(expected_ds, 6)
    assert set(summary["cross_model_estimates_usd"].keys()) == {
        "deepseek-v3",
        "qwen-plus",
        "gpt-4o-mini",
    }

    console = Console(record=True, width=100)
    tracker.print_summary(console)
    rendered = console.export_text()
    assert "Token & Cost Ledger" in rendered
    assert "DeepSeek-V3" in rendered
