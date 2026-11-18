"""Sliding-Window + Structured Summary Context Compressor without Tool-Call Orphaning."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from react_mcp_runtime.schema import Message

SUMMARY_MARKER = "[COMPRESSED_STATE_SUMMARY]"
SummarizerFn = Callable[[list[Message], str | None], Awaitable[str]]


def estimate_message_tokens(msg: Message) -> int:
    """Fast, deterministic token estimator (~4 ASCII chars/token + CJK weight + wire overhead)."""
    overhead = 6
    text_parts: list[str] = []
    if msg.content:
        text_parts.append(msg.content)
    if msg.name:
        text_parts.append(msg.name)
    if msg.tool_call_id:
        text_parts.append(msg.tool_call_id)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            overhead += 8
            text_parts.append(tc.id)
            text_parts.append(tc.function.name)
            text_parts.append(tc.function.arguments)

    combined = " ".join(text_parts)
    if not combined:
        return overhead
    ascii_chars = sum(1 for ch in combined if ord(ch) < 128)
    non_ascii_chars = len(combined) - ascii_chars
    return overhead + max(1, (ascii_chars + 3) // 4) + int(non_ascii_chars * 1.5)


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate total prompt tokens for a conversation history."""
    return 3 + sum(estimate_message_tokens(m) for m in messages)


def validate_tool_call_integrity(messages: list[Message]) -> None:
    """Assert OpenAI/MCP message invariant: zero orphaned `tool` or `tool_calls` turns."""
    idx = 0
    n = len(messages)
    while idx < n:
        msg = messages[idx]
        if msg.role == "assistant" and msg.tool_calls:
            expected_ids = [tc.id for tc in msg.tool_calls]
            idx += 1
            actual_ids: list[str] = []
            while idx < n and messages[idx].role == "tool":
                if messages[idx].tool_call_id is None:
                    raise ValueError("Found role='tool' message missing tool_call_id.")
                actual_ids.append(messages[idx].tool_call_id)
                idx += 1
            if set(expected_ids) != set(actual_ids) or len(expected_ids) != len(actual_ids):
                raise ValueError(
                    f"Broken assistant/tool pair: expected {expected_ids}, got {actual_ids}"
                )
            continue
        if msg.role == "tool":
            raise ValueError(
                f"Orphaned tool message detected (tool_call_id={msg.tool_call_id!r}) with no preceding assistant tool_calls."
            )
        idx += 1


def partition_atomic_groups(
    messages: list[Message],
) -> tuple[list[Message], Message | None, list[list[Message]]]:
    """Split history into (pinned_prefix, existing_summary_msg, atomic_turn_groups).

    Pinned prefix always keeps leading `system` messages and the initial `user` objective.
    Each atomic group is either:
      - `[assistant(tool_calls), tool_1, ..., tool_m]` (indivisible unit), or
      - `[single non-tool message]`.
    """
    idx = 0
    n = len(messages)
    pinned: list[Message] = []

    while idx < n and messages[idx].role == "system":
        if messages[idx].content and messages[idx].content.startswith(SUMMARY_MARKER):
            break
        pinned.append(messages[idx])
        idx += 1

    if idx < n and messages[idx].role == "user" and not (
        messages[idx].content and messages[idx].content.startswith(SUMMARY_MARKER)
    ):
        pinned.append(messages[idx])
        idx += 1

    existing_summary: Message | None = None
    if (
        idx < n
        and messages[idx].content
        and messages[idx].content.startswith(SUMMARY_MARKER)
    ):
        existing_summary = messages[idx]
        idx += 1

    groups: list[list[Message]] = []
    while idx < n:
        msg = messages[idx]
        if msg.role == "assistant" and msg.tool_calls:
            unit = [msg]
            expected_ids = {tc.id for tc in msg.tool_calls}
            idx += 1
            while idx < n and messages[idx].role == "tool":
                unit.append(messages[idx])
                expected_ids.discard(messages[idx].tool_call_id or "")
                idx += 1
            groups.append(unit)
        else:
            groups.append([msg])
            idx += 1

    return pinned, existing_summary, groups


def _compact_tool_output(tool_name: str, raw_content: str) -> str:
    """Extract high-signal facts from JSON or text tool outputs into a single compact line."""
    try:
        data: Any = json.loads(raw_content)
    except Exception:
        snippet = " ".join(raw_content.split())
        return f"{tool_name} -> {snippet[:140]}"

    if isinstance(data, dict):
        if data.get("ok") is False or data.get("isError") is True:
            err = data.get("error") or data.get("errors") or data.get("message") or "error"
            return f"{tool_name} [ERROR] -> {str(err)[:110]}"
        highlights: list[str] = []
        for key in (
            "service",
            "total_matched",
            "row_count",
            "commits",
            "top_exceptions",
            "rows",
            "revision",
            "tag_created",
            "status",
        ):
            if key not in data:
                continue
            val = data[key]
            if key == "commits" and isinstance(val, list) and val:
                top_commits = [f"{c.get('short_hash', c.get('hash', '')[:7])}:{c.get('subject', '')[:42]}" for c in val[:3]]
                highlights.append("commits=" + "; ".join(top_commits))
            elif key == "top_exceptions" and isinstance(val, list) and val:
                top_ex = [f"{e.get('exception_type')}({e.get('count')})" for e in val[:2]]
                highlights.append("exceptions=" + ", ".join(top_ex))
            elif key == "rows" and isinstance(val, list) and val:
                highlights.append(f"rows_sample={json.dumps(val[:2], ensure_ascii=False)[:140]}")
            else:
                highlights.append(f"{key}={json.dumps(val, ensure_ascii=False)[:80]}")
        if highlights:
            return f"{tool_name} -> " + ", ".join(highlights)
    snippet = " ".join(raw_content.split())
    return f"{tool_name} -> {snippet[:140]}"


def build_deterministic_summary(
    evicted_groups: list[list[Message]],
    prior_summary: str | None = None,
    max_summary_chars: int = 900,
) -> str:
    """Synthesize evicted atomic turn groups into a bounded structured state digest."""
    bullet_lines: list[str] = []
    if prior_summary:
        cleaned_prior = prior_summary.replace(SUMMARY_MARKER, "").strip()
        for line in cleaned_prior.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                bullet_lines.append(stripped)

    for group in evicted_groups:
        first = group[0]
        if first.role == "assistant" and first.tool_calls:
            tc_by_id = {tc.id: tc for tc in first.tool_calls}
            for tool_msg in group[1:]:
                tc = tc_by_id.get(tool_msg.tool_call_id or "")
                t_name = (tc.function.name if tc else tool_msg.name) or "unknown_tool"
                digest = _compact_tool_output(t_name, tool_msg.content or "")
                bullet_lines.append(f"- {digest}")
        elif first.content:
            snippet = " ".join(first.content.split())[:120]
            bullet_lines.append(f"- [{first.role}] {snippet}")

    # Deduplicate consecutive identical lines and bound total size
    deduped: list[str] = []
    for line in bullet_lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    body = "\n".join(deduped)
    if len(body) > max_summary_chars:
        # Keep most recent digest bullets within char budget
        kept: list[str] = []
        running = 0
        for line in reversed(deduped):
            if running + len(line) + 1 > max_summary_chars and kept:
                break
            kept.append(line)
            running += len(line) + 1
        kept.reverse()
        body = "- [older turns condensed]\n" + "\n".join(kept)

    return (
        f"{SUMMARY_MARKER}\n"
        "Earlier investigation turns were compressed to stay within context budget. Key verified facts:\n"
        f"{body}"
    )


class SlidingWindowCompressor:
    """Context compressor preserving system + goal + recent K atomic assistant/tool groups."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 3200,
        keep_recent_groups: int = 3,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.keep_recent_groups = max(1, keep_recent_groups)
        self.summarizer = summarizer

    async def compress_if_needed(
        self,
        messages: list[Message],
        *,
        overhead_tokens: int = 0,
    ) -> tuple[list[Message], bool]:
        """Return `(possibly_compressed_messages, did_compress)` while guaranteeing pair integrity."""
        current_tokens = estimate_tokens(messages) + max(0, overhead_tokens)
        if current_tokens <= self.max_context_tokens:
            return messages, False

        pinned, existing_summary, groups = partition_atomic_groups(messages)
        if len(groups) <= self.keep_recent_groups:
            return messages, False

        evicted_groups = groups[: -self.keep_recent_groups]
        kept_groups = groups[-self.keep_recent_groups :]
        prior_text = existing_summary.content if existing_summary else None

        if self.summarizer is not None:
            flat_evicted = [m for g in evicted_groups for m in g]
            summary_text = await self.summarizer(flat_evicted, prior_text)
            if not summary_text.startswith(SUMMARY_MARKER):
                summary_text = f"{SUMMARY_MARKER}\n{summary_text}"
        else:
            summary_text = build_deterministic_summary(evicted_groups, prior_text)

        summary_msg = Message(role="user", content=summary_text)
        rebuilt: list[Message] = [
            *pinned,
            summary_msg,
            *(msg for group in kept_groups for msg in group),
        ]
        validate_tool_call_integrity(rebuilt)
        return rebuilt, True
