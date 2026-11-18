"""SQL Read-Only AST/Token Guard, HITL Permission Interceptor & Hash-Chained Audit Log."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from react_mcp_runtime.schema import RiskLevel
from react_mcp_runtime.tools import ToolSecurityBlockedError

FORBIDDEN_SQL_TOKENS = frozenset(
    {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "REPLACE",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
        "REINDEX",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "LOAD_EXTENSION",
    }
)
ALLOWED_SQL_HEADS = frozenset({"SELECT", "WITH", "EXPLAIN"})

# Matches SQL comments (--... or /*...*/) OR single/double quoted string literals in one pass
_SQL_COMMENT_OR_LITERAL_RE = re.compile(
    r"(--[^\r\n]*)|(/\*[\s\S]*?\*/)|('(?:''|[^'])*')|(\"(?:\"\"|[^\"])*\")"
)
_SQL_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SQLReadOnlyGuard:
    """Lexical & structural SQL guard enforcing single-statement read-only SELECT/WITH queries."""

    def __init__(self, *, max_row_cap: int = 200) -> None:
        self.max_row_cap = max_row_cap

    @staticmethod
    def _sanitize_for_token_scan(sql: str) -> str:
        def _replacer(match: re.Match[str]) -> str:
            if match.group(1) or match.group(2):
                # Replace comments with whitespace so `SELECT/*x*/DROP` stays separated
                return " "
            # Replace string literals with benign placeholder so `'UPDATE_OK'` is allowed
            return "'__LITERAL__'"

        return _SQL_COMMENT_OR_LITERAL_RE.sub(_replacer, sql)

    def validate(self, sql: str, requested_max_rows: int = 50) -> tuple[str, int]:
        """Validate SQL is strictly single-statement read-only and clamp `max_rows`."""
        if not sql or not sql.strip():
            raise ToolSecurityBlockedError(
                "Empty SQL query rejected by SQLReadOnlyGuard.",
                reason_code="sql_empty",
            )

        # Check unclosed block comments
        if "/*" in sql and "*/" not in sql[sql.index("/*") + 2 :]:
            raise ToolSecurityBlockedError(
                "Unclosed SQL block comment rejected by SQLReadOnlyGuard.",
                reason_code="sql_malformed_comment",
            )

        sanitized = self._sanitize_for_token_scan(sql).strip()
        if not sanitized:
            raise ToolSecurityBlockedError(
                "Comment-only SQL statement rejected by SQLReadOnlyGuard.",
                reason_code="sql_empty",
            )

        # Allow at most one trailing semicolon; reject stacked multi-statement `;` injection
        without_trailing_semi = sanitized[:-1].strip() if sanitized.endswith(";") else sanitized
        if ";" in without_trailing_semi:
            raise ToolSecurityBlockedError(
                "Multi-statement SQL (`;` chaining) blocked by SQLReadOnlyGuard.",
                reason_code="sql_multi_statement",
            )

        tokens = [w.upper() for w in _SQL_WORD_RE.findall(without_trailing_semi) if w != "__LITERAL__"]
        if not tokens:
            raise ToolSecurityBlockedError(
                "No valid SQL keywords found.",
                reason_code="sql_invalid",
            )

        head = tokens[0]
        if head not in ALLOWED_SQL_HEADS:
            raise ToolSecurityBlockedError(
                f"Statement head `{head}` is forbidden; only read-only SELECT/WITH queries are permitted.",
                reason_code="sql_forbidden_statement",
            )

        for tok in tokens:
            if tok in FORBIDDEN_SQL_TOKENS:
                raise ToolSecurityBlockedError(
                    f"Forbidden SQL keyword `{tok}` intercepted by SQLReadOnlyGuard.",
                    reason_code=f"sql_forbidden_{tok.lower()}",
                )

        if head == "WITH" and "SELECT" not in tokens:
            raise ToolSecurityBlockedError(
                "CTE (`WITH`) query must select data via `SELECT`.",
                reason_code="sql_cte_without_select",
            )

        clamped_rows = max(1, min(int(requested_max_rows), self.max_row_cap))
        clean_query = sql.strip().rstrip(";").strip()
        return clean_query, clamped_rows


class PermissionPolicy(StrEnum):
    """MCP Client tool governance policies."""

    ALLOW_READ = "ALLOW_READ"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


@dataclass(frozen=True)
class ToolApprovalRequest:
    """Context presented to a human operator or automated reviewer callback."""

    tool_name: str
    risk_level: RiskLevel
    arguments: dict[str, Any]
    server_name: str = "dev-incident-mcp-server"


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome returned by a human-in-the-loop approval callback."""

    approved: bool
    reviewer: str = "human-sre-reviewer"
    reason: str = ""


ApprovalCallback = (
    Callable[[ToolApprovalRequest], ApprovalDecision]
    | Callable[[ToolApprovalRequest], Awaitable[ApprovalDecision]]
)


class ImmutableAuditLog:
    """Append-only JSONL security audit trail with SHA-256 cryptographic hash chaining."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._last_hash = self._load_tail_state()

    def _load_tail_state(self) -> tuple[int, str]:
        if not self.log_path.exists():
            return 0, "GENESIS"
        lines = [ln.strip() for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return 0, "GENESIS"
        last = json.loads(lines[-1])
        return int(last["seq"]), str(last["entry_hash"])

    @staticmethod
    def _compute_hash(record_without_hash: dict[str, Any]) -> str:
        canonical = json.dumps(record_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        tool_name: str,
        risk_level: RiskLevel,
        arguments: dict[str, Any],
        policy: str,
        decision: str,
        reviewer: str,
        reason: str = "",
    ) -> dict[str, Any]:
        self._seq += 1
        base_record: dict[str, Any] = {
            "seq": self._seq,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tool_name": tool_name,
            "risk_level": risk_level,
            "arguments": arguments,
            "policy": policy,
            "decision": decision,
            "reviewer": reviewer,
            "reason": reason,
            "prev_hash": self._last_hash,
        }
        entry_hash = self._compute_hash(base_record)
        full_record = {**base_record, "entry_hash": entry_hash}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(full_record, ensure_ascii=False) + "\n")
        self._last_hash = entry_hash
        return full_record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def verify_integrity(self) -> tuple[bool, int]:
        """Verify monotonic sequence numbers and SHA-256 hash chain across all entries."""
        entries = self.read_all()
        expected_prev = "GENESIS"
        for idx, entry in enumerate(entries, start=1):
            if entry.get("seq") != idx or entry.get("prev_hash") != expected_prev:
                return False, idx - 1
            recorded_hash = entry.get("entry_hash")
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if self._compute_hash(body) != recorded_hash:
                return False, idx - 1
            expected_prev = str(recorded_hash)
        return True, len(entries)


class HITLPermissionInterceptor:
    """Client-side MCP permission gate enforcing ALLOW_READ, REQUIRE_APPROVAL, and DENY."""

    def __init__(
        self,
        *,
        audit_log: ImmutableAuditLog,
        read_policy: PermissionPolicy = PermissionPolicy.ALLOW_READ,
        write_policy: PermissionPolicy = PermissionPolicy.REQUIRE_APPROVAL,
        tool_policies: dict[str, PermissionPolicy] | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.audit_log = audit_log
        self.read_policy = read_policy
        self.write_policy = write_policy
        self.tool_policies = dict(tool_policies or {})
        self.approval_callback = approval_callback

    def resolve_policy(self, tool_name: str, risk_level: RiskLevel) -> PermissionPolicy:
        if tool_name in self.tool_policies:
            return self.tool_policies[tool_name]
        return self.write_policy if risk_level == "write" else self.read_policy

    async def authorize(
        self,
        *,
        tool_name: str,
        risk_level: RiskLevel,
        arguments: dict[str, Any],
        server_name: str = "dev-incident-mcp-server",
    ) -> dict[str, Any]:
        policy = self.resolve_policy(tool_name, risk_level)

        if policy == PermissionPolicy.ALLOW_READ:
            if risk_level == "write":
                self.audit_log.append(
                    tool_name=tool_name,
                    risk_level=risk_level,
                    arguments=arguments,
                    policy=policy.value,
                    decision="denied_by_allow_read_only",
                    reviewer="policy-engine",
                    reason="Write tool disallowed under ALLOW_READ policy.",
                )
                raise ToolSecurityBlockedError(
                    f"Write tool `{tool_name}` blocked: active policy ALLOW_READ permits read-only tools only.",
                    reason_code="policy_allow_read_only",
                )
            return self.audit_log.append(
                tool_name=tool_name,
                risk_level=risk_level,
                arguments=arguments,
                policy=policy.value,
                decision="allowed",
                reviewer="policy-engine",
                reason="Read-only tool auto-allowed.",
            )

        if policy == PermissionPolicy.DENY:
            self.audit_log.append(
                tool_name=tool_name,
                risk_level=risk_level,
                arguments=arguments,
                policy=policy.value,
                decision="denied",
                reviewer="policy-engine",
                reason=f"Tool `{tool_name}` blocked by DENY policy.",
            )
            raise ToolSecurityBlockedError(
                f"Tool `{tool_name}` denied by client security policy (DENY).",
                reason_code="policy_denied",
            )

        # policy == PermissionPolicy.REQUIRE_APPROVAL
        if self.approval_callback is None:
            self.audit_log.append(
                tool_name=tool_name,
                risk_level=risk_level,
                arguments=arguments,
                policy=policy.value,
                decision="denied_no_approver",
                reviewer="policy-engine",
                reason="REQUIRE_APPROVAL configured without an interactive approval callback (fail-closed).",
            )
            raise ToolSecurityBlockedError(
                f"Tool `{tool_name}` requires human approval, but no approval_callback is registered.",
                reason_code="hitl_missing_callback",
            )

        req = ToolApprovalRequest(
            tool_name=tool_name,
            risk_level=risk_level,
            arguments=arguments,
            server_name=server_name,
        )
        raw_decision = self.approval_callback(req)
        decision = await raw_decision if inspect.isawaitable(raw_decision) else raw_decision

        if not decision.approved:
            self.audit_log.append(
                tool_name=tool_name,
                risk_level=risk_level,
                arguments=arguments,
                policy=policy.value,
                decision="rejected_by_human",
                reviewer=decision.reviewer,
                reason=decision.reason or "Operator declined high-risk action.",
            )
            raise ToolSecurityBlockedError(
                f"Human reviewer `{decision.reviewer}` rejected `{tool_name}`: {decision.reason or 'declined'}.",
                reason_code="hitl_rejected",
            )

        return self.audit_log.append(
            tool_name=tool_name,
            risk_level=risk_level,
            arguments=arguments,
            policy=policy.value,
            decision="approved_by_human",
            reviewer=decision.reviewer,
            reason=decision.reason or "Operator approved remediation.",
        )
