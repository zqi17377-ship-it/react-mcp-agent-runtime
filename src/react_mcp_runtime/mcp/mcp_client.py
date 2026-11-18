"""Async MCP Client supporting In-Memory & Stdio Subprocess Transports + HITL Interceptor."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from react_mcp_runtime.mcp.mcp_server import DevToolchainMCPServer
from react_mcp_runtime.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPResourceDescriptor,
    MCPToolDescriptor,
)
from react_mcp_runtime.mcp.security import HITLPermissionInterceptor
from react_mcp_runtime.schema import RiskLevel
from react_mcp_runtime.tools import ToolExecutionError, ToolRegistry, ToolSecurityBlockedError


class MCPTransport(ABC):
    """Wire transport abstraction for JSON-RPC 2.0 MCP communication."""

    @abstractmethod
    async def start(self) -> None:
        """Open transport channel."""

    @abstractmethod
    async def request(self, req: JsonRpcRequest) -> JsonRpcResponse:
        """Send JSON-RPC request frame and await correlated response."""

    @abstractmethod
    async def notify(self, note: JsonRpcNotification) -> None:
        """Send one-way JSON-RPC notification frame."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying resources / child process."""


class InMemoryMCPTransport(MCPTransport):
    """Zero-subprocess async wire transport exercising full JSON-RPC 2.0 serialization."""

    def __init__(self, server: DevToolchainMCPServer) -> None:
        self.server = server
        self._closed = False

    async def start(self) -> None:
        self._closed = False

    async def request(self, req: JsonRpcRequest) -> JsonRpcResponse:
        if self._closed:
            raise RuntimeError("InMemoryMCPTransport is closed.")
        wire_in = req.model_dump_json(exclude_none=True)
        wire_out = await self.server.handle_raw_message(wire_in)
        if wire_out is None:
            raise RuntimeError(f"MCP server returned empty frame for request id={req.id}")
        return JsonRpcResponse.model_validate_json(wire_out)

    async def notify(self, note: JsonRpcNotification) -> None:
        if self._closed:
            raise RuntimeError("InMemoryMCPTransport is closed.")
        wire_in = note.model_dump_json(exclude_none=True)
        await self.server.handle_raw_message(wire_in)

    async def close(self) -> None:
        self._closed = True


class StdioMCPTransport(MCPTransport):
    """Standard MCP stdio transport spawning `mcp_server.py` as an isolated child process."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        python_executable: str = sys.executable,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.python_executable = python_executable
        self.extra_env = dict(extra_env or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._io_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parents[2])
        existing_py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_py_path}" if existing_py_path else src_dir
        env.update(self.extra_env)

        def _spawn() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "react_mcp_runtime.mcp.mcp_server",
                    "--workspace",
                    str(self.workspace_root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

        self._proc = await asyncio.to_thread(_spawn)

    async def request(self, req: JsonRpcRequest) -> JsonRpcResponse:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("StdioMCPTransport has not been started.")
        proc = self._proc
        async with self._io_lock:
            wire_bytes = (req.model_dump_json(exclude_none=True) + "\n").encode("utf-8")

            def _exchange() -> tuple[bytes, bytes]:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(wire_bytes)
                proc.stdin.flush()
                line = proc.stdout.readline()
                err = b""
                if not line and proc.stderr is not None:
                    err = proc.stderr.read()
                return line, err

            raw_line, raw_err = await asyncio.wait_for(asyncio.to_thread(_exchange), timeout=15.0)
            if not raw_line:
                stderr_out = raw_err.decode("utf-8", errors="replace")
                raise RuntimeError(f"MCP stdio server closed stdout unexpectedly. stderr={stderr_out}")
            return JsonRpcResponse.model_validate_json(raw_line.decode("utf-8"))

    async def notify(self, note: JsonRpcNotification) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("StdioMCPTransport has not been started.")
        proc = self._proc
        async with self._io_lock:
            wire_bytes = (note.model_dump_json(exclude_none=True) + "\n").encode("utf-8")

            def _send_note() -> None:
                assert proc.stdin is not None
                proc.stdin.write(wire_bytes)
                proc.stdin.flush()

            await asyncio.to_thread(_send_note)

    async def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None

        def _shutdown() -> None:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
                proc.wait(timeout=3.0)
            finally:
                for stream in (proc.stdout, proc.stderr):
                    if stream and not stream.closed:
                        stream.close()

        await asyncio.to_thread(_shutdown)


class MCPClient:
    """Async MCP Client bridging JSON-RPC 2.0 servers into `ReActRuntime` with HITL security."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        security_interceptor: HITLPermissionInterceptor | None = None,
        client_name: str = "react-mcp-runtime-client",
        client_version: str = "0.1.0",
    ) -> None:
        self.transport = transport
        self.security_interceptor = security_interceptor
        self.client_name = client_name
        self.client_version = client_version
        self._req_id = 0
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self._cached_tools: dict[str, MCPToolDescriptor] = {}

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> dict[str, Any]:
        """Execute MCP `initialize` -> `notifications/initialized` JSON-RPC 2.0 handshake."""
        await self.transport.start()
        init_req = JsonRpcRequest(
            id=self._next_id(),
            method="initialize",
            params={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
        )
        resp = await self.transport.request(init_req)
        if resp.error is not None or resp.result is None:
            raise RuntimeError(f"MCP initialize failed: {resp.error}")
        self.server_info = resp.result.get("serverInfo", {})
        self.server_capabilities = resp.result.get("capabilities", {})

        await self.transport.notify(JsonRpcNotification(method="notifications/initialized"))
        return resp.result

    async def close(self) -> None:
        await self.transport.close()

    async def list_tools(self) -> list[MCPToolDescriptor]:
        resp = await self.transport.request(
            JsonRpcRequest(id=self._next_id(), method="tools/list", params={})
        )
        if resp.error is not None or resp.result is None:
            raise RuntimeError(f"MCP tools/list failed: {resp.error}")
        descriptors = [MCPToolDescriptor.model_validate(item) for item in resp.result.get("tools", [])]
        self._cached_tools = {d.name: d for d in descriptors}
        return descriptors

    async def list_resources(self) -> list[MCPResourceDescriptor]:
        resp = await self.transport.request(
            JsonRpcRequest(id=self._next_id(), method="resources/list", params={})
        )
        if resp.error is not None or resp.result is None:
            raise RuntimeError(f"MCP resources/list failed: {resp.error}")
        return [MCPResourceDescriptor.model_validate(r) for r in resp.result.get("resources", [])]

    async def read_resource(self, uri: str) -> list[dict[str, Any]]:
        resp = await self.transport.request(
            JsonRpcRequest(id=self._next_id(), method="resources/read", params={"uri": uri})
        )
        if resp.error is not None or resp.result is None:
            raise RuntimeError(f"MCP resources/read failed for `{uri}`: {resp.error}")
        return list(resp.result.get("contents", []))

    def _infer_risk_level(self, tool_name: str) -> RiskLevel:
        desc = self._cached_tools.get(tool_name)
        if desc is None:
            return "write"
        if not desc.annotations.readOnlyHint or desc.annotations.destructiveHint:
            return "write"
        return "read"

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Authorize via HITL interceptor, invoke MCP `tools/call`, and unpack text payload."""
        clean_args = {k: v for k, v in (arguments or {}).items() if v is not None}
        if not self._cached_tools:
            await self.list_tools()

        risk_level = self._infer_risk_level(name)
        if self.security_interceptor is not None:
            await self.security_interceptor.authorize(
                tool_name=name,
                risk_level=risk_level,
                arguments=clean_args,
                server_name=self.server_info.get("name", "mcp-server"),
            )

        resp = await self.transport.request(
            JsonRpcRequest(
                id=self._next_id(),
                method="tools/call",
                params={"name": name, "arguments": clean_args},
            )
        )
        if resp.error is not None or resp.result is None:
            raise ToolExecutionError(f"MCP tools/call RPC error: {resp.error}")

        content_blocks = resp.result.get("content", [])
        text_payload = content_blocks[0].get("text", "{}") if content_blocks else "{}"
        if resp.result.get("isError"):
            try:
                err_obj = json.loads(text_payload)
            except Exception:
                err_obj = {"error": text_payload}
            if isinstance(err_obj, dict) and err_obj.get("error_type") == "security_blocked":
                raise ToolSecurityBlockedError(
                    str(err_obj.get("error", "Blocked by server security guard")),
                    reason_code=str(err_obj.get("reason_code", "server_security_blocked")),
                )
        return str(text_payload)

    async def register_into(self, registry: ToolRegistry) -> list[str]:
        """Discover all MCP tools on the server and bind them into a `ReActRuntime` ToolRegistry."""
        tools = await self.list_tools()
        registered_names: list[str] = []

        for descriptor in tools:
            risk = self._infer_risk_level(descriptor.name)

            def _make_handler(t_name: str):
                async def _mcp_tool_proxy(**kwargs: Any) -> str:
                    return await self.call_tool(t_name, kwargs)

                return _mcp_tool_proxy

            registry.register_json_schema_tool(
                name=descriptor.name,
                description=descriptor.description,
                input_schema=descriptor.inputSchema,
                handler=_make_handler(descriptor.name),
                risk_level=risk,
            )
            registered_names.append(descriptor.name)

        return registered_names
