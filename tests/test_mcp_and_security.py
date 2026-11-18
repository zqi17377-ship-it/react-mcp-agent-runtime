"""Integration & security tests for MCP JSON-RPC 2.0 (Memory & Stdio), SQL Guard & HITL."""

from __future__ import annotations

import json

import pytest

from react_mcp_runtime.llm_client import MockLLMClient
from react_mcp_runtime.mcp import (
    MCP_PROTOCOL_VERSION,
    ApprovalDecision,
    DevToolchainMCPServer,
    HITLPermissionInterceptor,
    ImmutableAuditLog,
    IncidentWorkspace,
    InMemoryMCPTransport,
    MCPClient,
    PermissionPolicy,
    SQLReadOnlyGuard,
    StdioMCPTransport,
    ToolApprovalRequest,
)
from react_mcp_runtime.runtime import ReActRuntime
from react_mcp_runtime.schema import RuntimeConfig
from react_mcp_runtime.tools import ToolRegistry, ToolSecurityBlockedError


async def test_mcp_inmemory_handshake_tools_and_resources(incident_ws: IncidentWorkspace) -> None:
    server = DevToolchainMCPServer(incident_ws.root_dir)
    async with MCPClient(InMemoryMCPTransport(server)) as client:
        assert server.initialized is True
        assert client.server_info["name"] == "dev-incident-mcp-server"

        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "git_commit_history",
            "git_show_diff",
            "sqlite_readonly_query",
            "analyze_service_logs",
            "apply_incident_remediation",
        }

        resources = await client.list_resources()
        assert any(r.uri == "incident://runbook/sev2-checkout" for r in resources)
        contents = await client.read_resource("incident://runbook/sev2-checkout")
        assert "PoolExhaustedTimeoutError" in contents[0]["text"]

        # Verify raw protocol version in initialize response
        init_res = await client.connect()
        assert init_res["protocolVersion"] == MCP_PROTOCOL_VERSION


async def test_mcp_stdio_subprocess_transport_end_to_end(incident_ws: IncidentWorkspace) -> None:
    transport = StdioMCPTransport(incident_ws.root_dir)
    async with MCPClient(transport) as client:
        tools = await client.list_tools()
        assert len(tools) == 5

        git_raw = await client.call_tool("git_commit_history", {"limit": 6})
        git_data = json.loads(git_raw)
        assert git_data["ok"] is True
        assert git_data["count"] == 6

        sql_raw = await client.call_tool(
            "sqlite_readonly_query",
            {"sql": "SELECT service, version, p99_ms FROM latency_rollups ORDER BY p99_ms DESC LIMIT 2"},
        )
        sql_data = json.loads(sql_raw)
        assert sql_data["rows"][0]["version"] == "v2.14.3"
        assert sql_data["rows"][0]["p99_ms"] == 1840.0


async def test_git_history_diff_and_log_aggregation_accuracy(incident_ws: IncidentWorkspace) -> None:
    server = DevToolchainMCPServer(incident_ws.root_dir)
    async with MCPClient(InMemoryMCPTransport(server)) as client:
        diff_data = json.loads(await client.call_tool("git_show_diff", {"revision": "v2.14.3"}))
        assert diff_data["ok"] is True
        assert '"pool_max_size": 3' in diff_data["diff"]
        assert "blocking_sync_audit_flush" in diff_data["diff"]

        log_data = json.loads(
            await client.call_tool(
                "analyze_service_logs",
                {"service": "checkout-api", "min_level": "ERROR", "top_k": 3},
            )
        )
        assert log_data["total_matched"] == 50  # 42 PoolExhaustedTimeoutError + 8 CheckoutGatewayAbortError
        top_sig = log_data["top_exceptions"][0]
        assert top_sig["exception_type"] == "PoolExhaustedTimeoutError"
        assert top_sig["count"] == 42


async def test_sql_readonly_guard_blocks_all_write_and_injection_vectors(
    incident_ws: IncidentWorkspace,
) -> None:
    guard = SQLReadOnlyGuard(max_row_cap=100)

    # Legitimate SELECT with string literal containing 'UPDATE' keyword & CTE query
    clean_sql, cap = guard.validate(
        "WITH recent AS (SELECT * FROM deployments WHERE status != 'UPDATE_PENDING') SELECT * FROM recent;",
        requested_max_rows=500,
    )
    assert cap == 100
    assert clean_sql.startswith("WITH recent")

    malicious_queries = [
        "DROP TABLE deployments",
        "DELETE FROM error_events WHERE count > 10",
        "UPDATE deployments SET status = 'healthy'",
        "INSERT INTO deployments VALUES ('x','x','x','x','x','x','x')",
        "ALTER TABLE latency_rollups ADD COLUMN hacked INT",
        "ATTACH DATABASE ':memory:' AS pwn",
        "PRAGMA writable_schema = 1",
        "SELECT * FROM deployments; DROP TABLE error_events;",
        "/* bypass attempt */ DELETE FROM latency_rollups",
        "SELECT * FROM deployments -- comment\n; DELETE FROM deployments",
    ]
    for bad_sql in malicious_queries:
        with pytest.raises(ToolSecurityBlockedError):
            guard.validate(bad_sql)

    # End-to-end verification via MCP server: row truncation & SQLite untouched
    server = DevToolchainMCPServer(incident_ws.root_dir)
    async with MCPClient(InMemoryMCPTransport(server)) as client:
        res = json.loads(
            await client.call_tool(
                "sqlite_readonly_query",
                {"sql": "SELECT * FROM latency_rollups", "max_rows": 3},
            )
        )
        assert res["row_count"] == 3
        assert res["truncated"] is True

        with pytest.raises(ToolSecurityBlockedError):
            await client.call_tool(
                "sqlite_readonly_query",
                {"sql": "SELECT 1; DROP TABLE deployments;"},
            )
    assert incident_ws.verify_db_untouched() is True


async def test_hitl_interceptor_allow_read_deny_and_callback_approval_rejection(
    incident_ws: IncidentWorkspace,
) -> None:
    audit_log = ImmutableAuditLog(incident_ws.audit_log_path)
    server = DevToolchainMCPServer(incident_ws.root_dir)

    # 1. Policy DENY blocks write tools unconditionally
    deny_interceptor = HITLPermissionInterceptor(
        audit_log=audit_log,
        read_policy=PermissionPolicy.ALLOW_READ,
        write_policy=PermissionPolicy.DENY,
    )
    async with MCPClient(InMemoryMCPTransport(server), security_interceptor=deny_interceptor) as client:
        # Read tool succeeds
        logs = json.loads(await client.call_tool("analyze_service_logs", {"service": "checkout-api"}))
        assert logs["ok"] is True
        # Write tool is denied before touching git tags
        with pytest.raises(ToolSecurityBlockedError) as exc_info:
            await client.call_tool(
                "apply_incident_remediation",
                {
                    "action": "tag_rollback_target",
                    "target_commit": "v2.14.2",
                    "incident_id": "INC-DENIED",
                    "reason": "Testing DENY policy",
                },
            )
        assert exc_info.value.reason_code == "policy_denied"
    assert incident_ws.list_rollback_tags() == []

    # 2. Policy REQUIRE_APPROVAL with human rejection vs approval
    def reviewer_cb(req: ToolApprovalRequest) -> ApprovalDecision:
        if req.arguments.get("target_commit") == "v2.14.2":
            return ApprovalDecision(
                approved=True,
                reviewer="alice@sre",
                reason="Verified v2.14.2 is last healthy release",
            )
        return ApprovalDecision(
            approved=False,
            reviewer="alice@sre",
            reason="Rejected unverified rollback target",
        )

    hitl_interceptor = HITLPermissionInterceptor(
        audit_log=audit_log,
        write_policy=PermissionPolicy.REQUIRE_APPROVAL,
        approval_callback=reviewer_cb,
    )
    async with MCPClient(InMemoryMCPTransport(server), security_interceptor=hitl_interceptor) as client:
        # Rejected when pointing to wrong commit
        with pytest.raises(ToolSecurityBlockedError) as rej_exc:
            await client.call_tool(
                "apply_incident_remediation",
                {
                    "action": "tag_rollback_target",
                    "target_commit": "HEAD",
                    "incident_id": "INC-BAD",
                    "reason": "Premature tag",
                },
            )
        assert rej_exc.value.reason_code == "hitl_rejected"
        assert incident_ws.list_rollback_tags() == []

        # Approved when pointing to verified v2.14.2
        rem_res = json.loads(
            await client.call_tool(
                "apply_incident_remediation",
                {
                    "action": "tag_rollback_target",
                    "target_commit": "v2.14.2",
                    "incident_id": "INC-2026-0315",
                    "reason": "Rollback DB pool regression",
                },
            )
        )
        assert rem_res["ok"] is True
        assert rem_res["tag_created"] == "rollback/INC-2026-0315"
    assert "rollback/INC-2026-0315" in incident_ws.list_rollback_tags()


def test_immutable_audit_log_hash_chain_and_tamper_detection(ws_tmp_dir) -> None:
    log_file = ws_tmp_dir / "audit.jsonl"
    audit = ImmutableAuditLog(log_file)
    audit.append(
        tool_name="git_commit_history",
        risk_level="read",
        arguments={"limit": 6},
        policy="ALLOW_READ",
        decision="allowed",
        reviewer="policy-engine",
    )
    audit.append(
        tool_name="apply_incident_remediation",
        risk_level="write",
        arguments={"target_commit": "v2.14.2"},
        policy="REQUIRE_APPROVAL",
        decision="approved_by_human",
        reviewer="oncall-sre",
    )

    valid, count = audit.verify_integrity()
    assert valid is True and count == 2

    # Tamper with line 1 decision field and verify SHA-256 chain catches it immediately
    lines = log_file.read_text(encoding="utf-8").splitlines()
    tampered_first = json.loads(lines[0])
    tampered_first["decision"] = "forged_decision"
    lines[0] = json.dumps(tampered_first)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    valid_after, failed_idx = audit.verify_integrity()
    assert valid_after is False
    assert failed_idx == 0


async def test_end_to_end_react_agent_with_mcp_incident_triage(
    incident_ws: IncidentWorkspace,
) -> None:
    audit_log = ImmutableAuditLog(incident_ws.audit_log_path)
    interceptor = HITLPermissionInterceptor(
        audit_log=audit_log,
        write_policy=PermissionPolicy.REQUIRE_APPROVAL,
        approval_callback=lambda req: ApprovalDecision(
            approved=True,
            reviewer="sre-lead",
            reason="Confirmed v2.14.2 rollback target",
        ),
    )
    server = DevToolchainMCPServer(incident_ws.root_dir)
    registry = ToolRegistry()

    async with MCPClient(InMemoryMCPTransport(server), security_interceptor=interceptor) as mcp_client:
        await mcp_client.register_into(registry)
        runtime = ReActRuntime(
            llm_client=MockLLMClient(),
            tool_registry=registry,
            config=RuntimeConfig(max_steps=8, parallel_tool_calls=True),
        )
        result = await runtime.run(
            "Investigate checkout-api SEV-2 latency/error spike and tag rollback target [ACTION:REMEDIATE]."
        )

    assert result.status == "completed"
    assert "v2.14.3" in result.final_answer
    assert "PoolExhaustedTimeoutError" in result.final_answer
    assert "rollback/INC-2026-0315" in result.final_answer
    assert "rollback/INC-2026-0315" in incident_ws.list_rollback_tags()
    valid_chain, audit_count = audit_log.verify_integrity()
    assert valid_chain is True and audit_count >= 5
