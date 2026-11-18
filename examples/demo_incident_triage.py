"""End-to-End CLI Demo: Autonomous SEV-2 Incident Triage via ReAct Runtime + MCP + HITL Guard."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from react_mcp_runtime import (  # noqa: E402
    CostTracker,
    Message,
    ReActRuntime,
    RuntimeConfig,
    StepTrace,
    ToolRegistry,
    create_llm_client,
)
from react_mcp_runtime.mcp import (  # noqa: E402
    ApprovalDecision,
    DevToolchainMCPServer,
    HITLPermissionInterceptor,
    ImmutableAuditLog,
    InMemoryMCPTransport,
    MCPClient,
    PermissionPolicy,
    StdioMCPTransport,
    ToolApprovalRequest,
    build_incident_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReAct + MCP Production Incident Triage Demo"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "memory"],
        default="stdio",
        help="MCP wire transport mode (default: real child-process stdio)",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "mock", "openai"],
        default="auto",
        help="LLM backend selection (default: auto -> OpenAI if OPENAI_API_KEY set else mock)",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "reject", "interactive"],
        default="approve",
        help="HITL decision mode for high-risk write remediation (default: approve)",
    )
    parser.add_argument(
        "--inject-json-fault",
        action="store_true",
        help="Inject a malformed tool-call JSON on Step 1 to demonstrate live self-healing",
    )
    return parser.parse_args()


async def run_demo(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    console = Console(width=104)

    demo_ws_dir = PROJECT_ROOT / ".tmp_workspace" / "demo_incident"
    ws = build_incident_workspace(demo_ws_dir, overwrite=True)

    console.print(
        Panel.fit(
            f"[bold cyan]Zero-Framework Async ReAct + MCP Incident Triage Demo[/bold cyan]\n"
            f"• Workspace: [dim]{ws.root_dir}[/dim]\n"
            f"• MCP Transport: [bold yellow]{args.transport}[/bold yellow] (JSON-RPC 2.0 `2024-11-05`)\n"
            f"• Write Policy: [bold magenta]REQUIRE_APPROVAL[/bold magenta] + SHA-256 Audit Trail",
            border_style="cyan",
        )
    )

    def hitl_callback(req: ToolApprovalRequest) -> ApprovalDecision:
        req_table = Table(show_header=False, box=None, padding=(0, 1))
        req_table.add_row("[bold]Tool:[/bold]", f"[red]{req.tool_name}[/red] (risk={req.risk_level})")
        req_table.add_row("[bold]Server:[/bold]", req.server_name)
        for k, v in req.arguments.items():
            req_table.add_row(f"[bold]{k}:[/bold]", str(v))
        console.print(
            Panel(
                req_table,
                title="[bold yellow]HITL Security Gate — High-Risk Write Intercepted[/bold yellow]",
                border_style="yellow",
            )
        )

        if args.decision == "interactive" and sys.stdin.isatty():
            ans = console.input("[bold yellow]Approve remediation action? [y/N]: [/bold yellow]").strip().lower()
            approved = ans in ("y", "yes")
        else:
            approved = args.decision == "approve"

        verdict_label = "[bold green]APPROVED[/bold green]" if approved else "[bold red]REJECTED[/bold red]"
        console.print(f"  -> Operator Verdict: {verdict_label}\n")
        return ApprovalDecision(
            approved=approved,
            reviewer="oncall-sre@commerce.internal",
            reason=(
                "Verified v2.14.3 DB pool regression; approved rollback tag to v2.14.2"
                if approved
                else "Operator declined write action during dry-run"
            ),
        )

    audit_log = ImmutableAuditLog(ws.audit_log_path)
    interceptor = HITLPermissionInterceptor(
        audit_log=audit_log,
        read_policy=PermissionPolicy.ALLOW_READ,
        write_policy=PermissionPolicy.REQUIRE_APPROVAL,
        approval_callback=hitl_callback,
    )

    transport = (
        StdioMCPTransport(ws.root_dir)
        if args.transport == "stdio"
        else InMemoryMCPTransport(DevToolchainMCPServer(ws.root_dir))
    )

    def on_step(trace: StepTrace, assistant_msg: Message) -> None:
        tools_str = ", ".join(trace.tool_names) if trace.tool_names else "Final Synthesis"
        badges: list[str] = []
        if len(trace.tool_names) > 1:
            badges.append("[cyan]PARALLEL[/cyan]")
        if trace.self_healed_errors:
            badges.append(f"[yellow]SELF-HEALED x{trace.self_healed_errors}[/yellow]")
        if trace.security_blocks:
            badges.append(f"[red]SECURITY-BLOCKED x{trace.security_blocks}[/red]")
        if trace.compressed_before_step:
            badges.append("[magenta]COMPRESSED[/magenta]")
        badge_suffix = (" " + " ".join(badges)) if badges else ""
        console.print(
            f"[bold blue][Step {trace.step}][/bold blue] {tools_str}{badge_suffix} "
            f"[dim]({trace.prompt_tokens}p/{trace.completion_tokens}c tok, {trace.latency_ms:.1f} ms)[/dim]"
        )
        if trace.thought and trace.tool_names:
            console.print(f"  [dim]Thought:[/dim] {trace.thought}")

    registry = ToolRegistry()
    cost_tracker = CostTracker()
    llm_client = create_llm_client(mode=args.llm_mode)

    fault_tag = " [FAULT:TRUNCATED_JSON]" if args.inject_json_fault else ""
    prompt = (
        "SEV-2 Alert on `checkout-api`: p99 latency and HTTP 503 error rate spiked after the 14:18 UTC deployment. "
        "Correlate structured service error logs, SQLite `deployments`/`latency_rollups`, and Git commit diffs "
        "to pinpoint the exact root-cause commit and last healthy version, then apply a rollback target tag "
        f"for incident `INC-2026-0315`. [ACTION:REMEDIATE]{fault_tag}"
    )

    async with MCPClient(transport, security_interceptor=interceptor) as mcp_client:
        bound_tools = await mcp_client.register_into(registry)
        console.print(f"[green]✓[/green] Connected to MCP server; bound {len(bound_tools)} tools: {', '.join(bound_tools)}\n")

        runtime = ReActRuntime(
            llm_client=llm_client,
            tool_registry=registry,
            config=RuntimeConfig(
                max_steps=10,
                max_context_tokens=3200,
                keep_recent_groups=3,
                max_self_heal_retries=3,
                parallel_tool_calls=True,
            ),
            cost_tracker=cost_tracker,
            on_step=on_step,
        )
        result = await runtime.run(prompt)

    console.print()
    console.print(
        Panel(
            Markdown(result.final_answer),
            title=f"[bold green]Incident Triage Report (status={result.status}, steps={result.steps})[/bold green]",
            border_style="green",
        )
    )

    cost_tracker.print_summary(console)

    chain_ok, entry_count = audit_log.verify_integrity()
    rollback_tags = ws.list_rollback_tags()
    console.print(
        f"[bold]Security Audit Trail:[/bold] {entry_count} hash-chained JSONL entries | "
        f"SHA-256 Integrity={'[green]VERIFIED[/green]' if chain_ok else '[red]BROKEN[/red]'} | "
        f"Git Rollback Tags={rollback_tags}"
    )
    return 0 if result.status == "completed" and chain_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_demo(parse_args())))
