"""Custom MCP Dev Toolchain Server (JSON-RPC 2.0 over In-Memory & Stdio Transports)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiosqlite

from react_mcp_runtime.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MCP_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
    JsonRpcResponse,
    MCPResourceDescriptor,
    MCPToolAnnotations,
    MCPToolDescriptor,
)
from react_mcp_runtime.mcp.security import SQLReadOnlyGuard
from react_mcp_runtime.tools import ToolSecurityBlockedError

SEVERITY_RANK = {"INFO": 10, "WARN": 20, "ERROR": 30, "CRITICAL": 40}
SAFE_REV_RE = re.compile(r"^[A-Za-z0-9_./~^-]{1,80}$")


class DevToolchainMCPServer:
    """MCP 2024-11-05 compliant server exposing Git, Read-Only SQLite, Log Analytics & Remediation."""

    SERVER_NAME = "dev-incident-mcp-server"
    SERVER_VERSION = "0.1.0"

    def __init__(
        self,
        workspace_root: Path,
        *,
        enable_sql_guard: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.repo_dir = self.workspace_root / "repo"
        self.db_path = self.workspace_root / "service_metrics.db"
        self.log_path = self.workspace_root / "logs" / "service_errors.jsonl"
        self.remediation_journal_path = self.workspace_root / "remediation_journal.jsonl"
        self.enable_sql_guard = enable_sql_guard
        self.sql_guard = SQLReadOnlyGuard(max_row_cap=200)
        self.initialized = False

    def tool_descriptors(self) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name="git_commit_history",
                description="Query recent Git commits in the incident repository asynchronously.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 6,
                            "description": "Maximum number of commits to retrieve.",
                        },
                        "path_filter": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": "Optional relative file/folder path filter.",
                        },
                    },
                    "additionalProperties": False,
                },
                annotations=MCPToolAnnotations(readOnlyHint=True, destructiveHint=False),
            ),
            MCPToolDescriptor(
                name="git_show_diff",
                description="Inspect commit metadata and unified patch diff for a Git commit or tag.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "revision": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Commit SHA, short SHA, branch, or release tag (e.g. v2.14.3).",
                        }
                    },
                    "required": ["revision"],
                    "additionalProperties": False,
                },
                annotations=MCPToolAnnotations(readOnlyHint=True, destructiveHint=False),
            ),
            MCPToolDescriptor(
                name="sqlite_readonly_query",
                description=(
                    "Execute a read-only SQL SELECT query against `service_metrics.db` "
                    "(tables: `deployments`, `latency_rollups`, `error_events`)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Single read-only SELECT/WITH SQL statement.",
                        },
                        "max_rows": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 50,
                            "description": "Maximum number of rows to return.",
                        },
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
                annotations=MCPToolAnnotations(readOnlyHint=True, destructiveHint=False),
            ),
            MCPToolDescriptor(
                name="analyze_service_logs",
                description=(
                    "Filter and aggregate structured JSONL service error logs by service, minimum severity, "
                    "time window, and top exception stacktrace signatures."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": "Optional service filter (e.g. checkout-api, catalog-api).",
                        },
                        "min_level": {
                            "type": "string",
                            "enum": ["INFO", "WARN", "ERROR", "CRITICAL"],
                            "default": "ERROR",
                            "description": "Minimum severity threshold.",
                        },
                        "start_time": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": "Optional ISO-8601 UTC lower bound.",
                        },
                        "end_time": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": "Optional ISO-8601 UTC upper bound.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "Number of top exception signatures to return.",
                        },
                    },
                    "additionalProperties": False,
                },
                annotations=MCPToolAnnotations(readOnlyHint=True, destructiveHint=False),
            ),
            MCPToolDescriptor(
                name="apply_incident_remediation",
                description=(
                    "HIGH-RISK WRITE TOOL: Create an incident rollback/quarantine Git tag and append "
                    "an entry to the remediation journal. Requires HITL approval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["tag_rollback_target", "quarantine_release"],
                            "description": "Remediation action type.",
                        },
                        "target_commit": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Target Git commit or tag for rollback/quarantine.",
                        },
                        "incident_id": {
                            "type": "string",
                            "minLength": 3,
                            "description": "Incident identifier (e.g. INC-2026-0315).",
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 5,
                            "description": "Root-cause justification for the remediation.",
                        },
                    },
                    "required": ["action", "target_commit", "incident_id", "reason"],
                    "additionalProperties": False,
                },
                annotations=MCPToolAnnotations(readOnlyHint=False, destructiveHint=True),
            ),
        ]

    def resource_descriptors(self) -> list[MCPResourceDescriptor]:
        return [
            MCPResourceDescriptor(
                uri="incident://runbook/sev2-checkout",
                name="Checkout-API SEV-2 Triage Runbook",
                description="Standard operating procedure for triaging checkout-api latency & 5xx incidents.",
                mimeType="text/markdown",
            ),
            MCPResourceDescriptor(
                uri="incident://workspace/schema",
                name="Metrics SQLite Schema & Log Manifest",
                description="Table schemas and file locations inside the incident workspace.",
                mimeType="application/json",
            ),
        ]

    async def handle_raw_message(self, raw_line: str) -> str | None:
        """Process one newline-delimited JSON-RPC 2.0 frame; returns `None` for notifications."""
        text = raw_line.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return JsonRpcResponse(
                id=None,
                error=JsonRpcError(code=PARSE_ERROR, message=f"Parse error: {exc.msg}"),
            ).to_wire_json()

        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "method" not in payload:
            return JsonRpcResponse(
                id=payload.get("id") if isinstance(payload, dict) else None,
                error=JsonRpcError(code=INVALID_REQUEST, message="Invalid JSON-RPC 2.0 envelope."),
            ).to_wire_json()

        method = str(payload["method"])
        params = payload.get("params") or {}
        is_notification = "id" not in payload
        req_id = payload.get("id")

        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        try:
            result = await self._dispatch_method(method, params)
            return JsonRpcResponse(id=req_id, result=result).to_wire_json()
        except KeyError as exc:
            return JsonRpcResponse(
                id=req_id,
                error=JsonRpcError(code=METHOD_NOT_FOUND, message=f"Method not found: {exc}"),
            ).to_wire_json()
        except ValueError as exc:
            return JsonRpcResponse(
                id=req_id,
                error=JsonRpcError(code=INVALID_PARAMS, message=str(exc)),
            ).to_wire_json()
        except Exception as exc:
            return JsonRpcResponse(
                id=req_id,
                error=JsonRpcError(code=INTERNAL_ERROR, message=f"{type(exc).__name__}: {exc}"),
            ).to_wire_json()

    async def _dispatch_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
                "serverInfo": {
                    "name": self.SERVER_NAME,
                    "version": self.SERVER_VERSION,
                },
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [t.model_dump() for t in self.tool_descriptors()]}
        if method == "tools/call":
            tool_name = params.get("name")
            if not tool_name:
                raise ValueError("Missing `name` in tools/call params.")
            arguments = params.get("arguments") or {}
            return await self._call_tool(str(tool_name), arguments)
        if method == "resources/list":
            return {"resources": [r.model_dump() for r in self.resource_descriptors()]}
        if method == "resources/read":
            uri = params.get("uri")
            if not uri:
                raise ValueError("Missing `uri` in resources/read params.")
            return self._read_resource(str(uri))
        raise KeyError(method)

    def _read_resource(self, uri: str) -> dict[str, Any]:
        if uri == "incident://runbook/sev2-checkout":
            runbook_file = self.repo_dir / "docs" / "runbook.md"
            text = runbook_file.read_text(encoding="utf-8") if runbook_file.exists() else "# Runbook unavailable"
            return {
                "contents": [
                    {"uri": uri, "mimeType": "text/markdown", "text": text}
                ]
            }
        if uri == "incident://workspace/schema":
            manifest = {
                "sqlite_tables": {
                    "deployments": ["deploy_id", "service", "version", "git_commit", "deployed_at", "deployed_by", "status"],
                    "latency_rollups": ["window_start", "service", "version", "req_count", "p50_ms", "p95_ms", "p99_ms", "error_rate"],
                    "error_events": ["event_id", "occurred_at", "service", "version", "error_code", "exception_type", "count"],
                },
                "jsonl_log_file": str(self.log_path.name),
            }
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(manifest, indent=2),
                    }
                ]
            }
        raise ValueError(f"Unknown resource URI: {uri}")

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "git_commit_history":
                data = await self._tool_git_commit_history(
                    limit=int(args.get("limit") or 6),
                    path_filter=args.get("path_filter"),
                )
            elif name == "git_show_diff":
                data = await self._tool_git_show_diff(revision=str(args["revision"]))
            elif name == "sqlite_readonly_query":
                data = await self._tool_sqlite_readonly_query(
                    sql=str(args["sql"]),
                    max_rows=int(args.get("max_rows") or 50),
                )
            elif name == "analyze_service_logs":
                data = await self._tool_analyze_service_logs(
                    service=args.get("service"),
                    min_level=str(args.get("min_level") or "ERROR"),
                    start_time=args.get("start_time"),
                    end_time=args.get("end_time"),
                    top_k=int(args.get("top_k") or 5),
                )
            elif name == "apply_incident_remediation":
                data = await self._tool_apply_incident_remediation(
                    action=str(args["action"]),
                    target_commit=str(args["target_commit"]),
                    incident_id=str(args["incident_id"]),
                    reason=str(args["reason"]),
                )
            else:
                raise ValueError(f"Unknown MCP tool: {name}")

            return {
                "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
                "isError": False,
            }
        except ToolSecurityBlockedError as exc:
            err_body = {
                "ok": False,
                "error_type": "security_blocked",
                "reason_code": exc.reason_code,
                "tool": name,
                "error": str(exc),
            }
            return {
                "content": [{"type": "text", "text": json.dumps(err_body, ensure_ascii=False)}],
                "isError": True,
            }
        except Exception as exc:
            err_body = {
                "ok": False,
                "error_type": "tool_execution_error",
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return {
                "content": [{"type": "text", "text": json.dumps(err_body, ensure_ascii=False)}],
                "isError": True,
            }

    async def _run_git_async(self, git_args: list[str]) -> str:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"

        def _exec_git() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                ["git", *git_args],
                cwd=str(self.repo_dir),
                env=env,
                capture_output=True,
                check=False,
            )

        proc = await asyncio.to_thread(_exec_git)
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git {' '.join(git_args[:2])} failed: {err_msg}")
        return proc.stdout.decode("utf-8", errors="replace")

    async def _tool_git_commit_history(
        self,
        *,
        limit: int = 6,
        path_filter: str | None = None,
    ) -> dict[str, Any]:
        clamped = max(1, min(limit, 50))
        sep = "\x1f"
        rec_sep = "\x1e"
        fmt = f"%H{sep}%h{sep}%an{sep}%aI{sep}%D{sep}%s{rec_sep}"
        cmd = ["log", f"-n{clamped}", f"--pretty=format:{fmt}", "--name-only"]
        if path_filter:
            if path_filter.startswith("-"):
                raise ValueError("Invalid path_filter starting with '-'.")
            cmd.extend(["--", path_filter])

        raw = await self._run_git_async(cmd)
        commits: list[dict[str, Any]] = []
        for block in raw.split("\n"):
            pass
        # Parse records separated by rec_sep
        chunks = raw.split(rec_sep)
        # Format emits `<header>\x1e\n<files>\n<next_header>\x1e...`
        pending_header: list[str] | None = None
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                header_line = chunk.strip()
                if header_line:
                    pending_header = header_line.split(sep)
                continue
            lines = [ln.strip() for ln in chunk.strip().splitlines() if ln.strip()]
            files: list[str] = []
            next_header: list[str] | None = None
            for ln in lines:
                if sep in ln:
                    next_header = ln.split(sep)
                else:
                    files.append(ln)
            if pending_header and len(pending_header) >= 6:
                commits.append(
                    {
                        "hash": pending_header[0],
                        "short_hash": pending_header[1],
                        "author": pending_header[2],
                        "date": pending_header[3],
                        "refs": pending_header[4],
                        "subject": pending_header[5],
                        "files": files,
                    }
                )
            pending_header = next_header

        return {"ok": True, "count": len(commits), "commits": commits}

    async def _tool_git_show_diff(self, *, revision: str) -> dict[str, Any]:
        rev = revision.strip()
        if not SAFE_REV_RE.match(rev) or rev.startswith("-"):
            raise ToolSecurityBlockedError(
                f"Unsafe Git revision specifier `{rev}` rejected.",
                reason_code="git_invalid_revision",
            )
        diff_out = await self._run_git_async(["show", "--stat", "--patch", "--no-color", rev])
        return {
            "ok": True,
            "revision": rev,
            "diff": diff_out[:3600],
            "truncated": len(diff_out) > 3600,
        }

    async def _tool_sqlite_readonly_query(
        self,
        *,
        sql: str,
        max_rows: int = 50,
    ) -> dict[str, Any]:
        if self.enable_sql_guard:
            clean_sql, row_cap = self.sql_guard.validate(sql, max_rows)
            posix_db = self.db_path.as_posix()
            encoded_path = quote(posix_db, safe="/:")
            uri = f"file:{encoded_path}?mode=ro"
            async with aiosqlite.connect(uri, uri=True) as conn:
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA query_only = ON;")
                async with conn.execute(clean_sql) as cursor:
                    fetched = await cursor.fetchmany(row_cap + 1)
                    col_names = [desc[0] for desc in (cursor.description or [])]
        else:
            # Unguarded mode used strictly by eval baseline in disposable temp dirs
            row_cap = max(1, min(max_rows, 200))
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                if ";" in sql.strip().rstrip(";"):
                    await conn.executescript(sql)
                    await conn.commit()
                    return {"ok": True, "columns": [], "rows": [], "row_count": 0, "truncated": False}
                async with conn.execute(sql) as cursor:
                    fetched = await cursor.fetchmany(row_cap + 1)
                    col_names = [desc[0] for desc in (cursor.description or [])]
                await conn.commit()

        truncated = len(fetched) > row_cap
        sliced = fetched[:row_cap]
        rows = [dict(zip(col_names, tuple(r), strict=False)) for r in sliced]
        return {
            "ok": True,
            "columns": col_names,
            "row_count": len(rows),
            "truncated": truncated,
            "rows": rows,
        }

    async def _tool_analyze_service_logs(
        self,
        *,
        service: str | None = None,
        min_level: str = "ERROR",
        start_time: str | None = None,
        end_time: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        threshold = SEVERITY_RANK.get(min_level.upper(), 30)
        level_counts: Counter[str] = Counter()
        service_version_counts: Counter[str] = Counter()
        sig_groups: dict[tuple[str, str], dict[str, Any]] = {}
        total_matched = 0

        if self.log_path.exists():
            lines = await asyncio.to_thread(self.log_path.read_text, "utf-8")
            for raw_ln in lines.splitlines():
                if not raw_ln.strip():
                    continue
                rec = json.loads(raw_ln)
                lvl = str(rec.get("level", "INFO")).upper()
                if SEVERITY_RANK.get(lvl, 0) < threshold:
                    continue
                svc = str(rec.get("service", ""))
                if service and svc != service:
                    continue
                ts = str(rec.get("timestamp", ""))
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue

                total_matched += 1
                level_counts[lvl] += 1
                ver = str(rec.get("version", "unknown"))
                service_version_counts[f"{svc}@{ver}"] += 1

                exc_type = str(rec.get("exception_type", "None"))
                sig = str(rec.get("stacktrace_signature", ""))
                key = (exc_type, sig)
                bucket = sig_groups.get(key)
                if bucket is None:
                    sig_groups[key] = {
                        "exception_type": exc_type,
                        "stacktrace_signature": sig,
                        "count": 1,
                        "sample_message": rec.get("message", ""),
                        "first_seen": ts,
                        "last_seen": ts,
                    }
                else:
                    bucket["count"] += 1
                    bucket["last_seen"] = max(bucket["last_seen"], ts)

        ranked = sorted(sig_groups.values(), key=lambda x: x["count"], reverse=True)[: max(1, min(top_k, 20))]
        return {
            "ok": True,
            "service_filter": service,
            "min_level": min_level.upper(),
            "total_matched": total_matched,
            "by_level": dict(level_counts),
            "by_service_version": dict(service_version_counts),
            "top_exceptions": ranked,
        }

    async def _tool_apply_incident_remediation(
        self,
        *,
        action: str,
        target_commit: str,
        incident_id: str,
        reason: str,
    ) -> dict[str, Any]:
        rev = target_commit.strip()
        if not SAFE_REV_RE.match(rev) or rev.startswith("-"):
            raise ValueError(f"Invalid target_commit `{rev}`.")
        safe_inc = re.sub(r"[^A-Za-z0-9_.-]", "-", incident_id.strip())
        resolved_sha = (await self._run_git_async(["rev-parse", "--verify", rev])).strip()
        prefix = "rollback" if action == "tag_rollback_target" else "quarantine"
        tag_name = f"{prefix}/{safe_inc}"
        await self._run_git_async(["tag", "-f", tag_name, resolved_sha])

        journal_entry = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "action": action,
            "incident_id": safe_inc,
            "target_commit": resolved_sha,
            "tag_created": tag_name,
            "reason": reason,
        }
        with self.remediation_journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(journal_entry, ensure_ascii=False) + "\n")

        return {
            "ok": True,
            "status": "remediation_applied",
            **journal_entry,
        }


async def run_stdio_server(workspace_root: Path) -> None:
    """Serve MCP JSON-RPC 2.0 frames over stdin/stdout (cross-platform Windows/POSIX safe)."""
    server = DevToolchainMCPServer(workspace_root=workspace_root)
    while True:
        raw_bytes = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not raw_bytes:
            break
        line = raw_bytes.decode("utf-8", errors="replace")
        response_json = await server.handle_raw_message(line)
        if response_json is not None:
            out_bytes = (response_json + "\n").encode("utf-8")
            sys.stdout.buffer.write(out_bytes)
            sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev Incident Toolchain MCP Server (stdio)")
    parser.add_argument("--workspace", type=Path, required=True, help="Path to incident workspace root")
    args = parser.parse_args()
    asyncio.run(run_stdio_server(args.workspace))


if __name__ == "__main__":
    main()
