"""Real-Time Token, Latency, Retry & Multi-Model Cost Ledger with Rich CLI Rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from react_mcp_runtime.schema import LLMResponse

# Pricing in USD per 1,000,000 tokens: (input_usd_per_1m, output_usd_per_1m)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "deepseek-v3": (0.27, 1.10),
    "qwen-plus": (0.40, 1.20),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mock-react-v1": (0.27, 1.10),  # Reference-priced at DeepSeek-V3 rates for offline planning
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated USD cost for a given model and token split."""
    normalized = model.lower().strip()
    rates = MODEL_PRICING_PER_1M.get(normalized)
    if rates is None:
        for key, candidate in MODEL_PRICING_PER_1M.items():
            if key in normalized:
                rates = candidate
                break
    if rates is None:
        rates = MODEL_PRICING_PER_1M["gpt-4o-mini"]
    in_rate, out_rate = rates
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


@dataclass
class CallLedgerEntry:
    """Single LLM completion accounting record."""

    step: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retries: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates token usage, latency distribution, retry count, and model cost estimates."""

    entries: list[CallLedgerEntry] = field(default_factory=list)

    def record(self, step: int, response: LLMResponse) -> CallLedgerEntry:
        cost = compute_cost_usd(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        entry = CallLedgerEntry(
            step=step,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=round(response.latency_ms, 2),
            retries=response.retries,
            cost_usd=cost,
        )
        self.entries.append(entry)
        return entry

    @property
    def prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.entries)

    @property
    def completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_latency_ms(self) -> float:
        return round(sum(e.latency_ms for e in self.entries), 2)

    @property
    def p95_latency_ms(self) -> float:
        if not self.entries:
            return 0.0
        sorted_lat = sorted(e.latency_ms for e in self.entries)
        idx = min(len(sorted_lat) - 1, max(0, int(len(sorted_lat) * 0.95 + 0.999) - 1))
        return round(sorted_lat[idx], 2)

    @property
    def total_retries(self) -> int:
        return sum(e.retries for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def summary_dict(self) -> dict[str, Any]:
        p_tok = self.prompt_tokens
        c_tok = self.completion_tokens
        active_model = self.entries[-1].model if self.entries else "mock-react-v1"
        return {
            "llm_calls": len(self.entries),
            "active_model": active_model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
            "peak_prompt_tokens": max((e.prompt_tokens for e in self.entries), default=0),
            "total_latency_ms": self.total_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "retry_count": self.total_retries,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
            "cross_model_estimates_usd": {
                "deepseek-v3": round(compute_cost_usd("deepseek-v3", p_tok, c_tok), 6),
                "qwen-plus": round(compute_cost_usd("qwen-plus", p_tok, c_tok), 6),
                "gpt-4o-mini": round(compute_cost_usd("gpt-4o-mini", p_tok, c_tok), 6),
            },
        }

    def build_rich_panel(self, title: str = "Token & Cost Ledger") -> Panel:
        data = self.summary_dict()
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Cross-Model CostProjection", justify="right")

        cross = data["cross_model_estimates_usd"]
        table.add_row(
            "LLM Calls / Retries",
            f"{data['llm_calls']} calls / {data['retry_count']} retries",
            f"DeepSeek-V3: ${cross['deepseek-v3']:.6f}",
        )
        table.add_row(
            "Prompt / Peak Tokens",
            f"{data['prompt_tokens']:,} / {data['peak_prompt_tokens']:,}",
            f"Qwen-Plus:   ${cross['qwen-plus']:.6f}",
        )
        table.add_row(
            "Completion / Total Tokens",
            f"{data['completion_tokens']:,} / {data['total_tokens']:,}",
            f"GPT-4o-mini: ${cross['gpt-4o-mini']:.6f}",
        )
        table.add_row(
            "Total / P95 Latency",
            f"{data['total_latency_ms']:.1f} ms / {data['p95_latency_ms']:.1f} ms",
            f"Active ({data['active_model']}): ${data['estimated_cost_usd']:.6f}",
        )
        return Panel(table, title=title, border_style="green")

    def print_summary(self, console: Console | None = None) -> None:
        target = console or Console()
        target.print(self.build_rich_panel())
