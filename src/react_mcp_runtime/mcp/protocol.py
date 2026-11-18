"""Model Context Protocol (MCP) JSON-RPC 2.0 Wire Protocol Models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MCP_PROTOCOL_VERSION = "2024-11-05"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(BaseModel):
    """Standard JSON-RPC 2.0 error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Any = None


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request requiring a correlated response."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] | None = None


class JsonRpcNotification(BaseModel):
    """JSON-RPC 2.0 one-way notification (no `id` field)."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response envelope."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None

    def to_wire_json(self) -> str:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            err_dict: dict[str, Any] = {
                "code": self.error.code,
                "message": self.error.message,
            }
            if self.error.data is not None:
                err_dict["data"] = self.error.data
            payload["error"] = err_dict
        else:
            payload["result"] = self.result if self.result is not None else {}
        return JsonRpcResponse.model_validate(payload).model_dump_json(exclude_none=True)


class MCPToolAnnotations(BaseModel):
    """MCP tool safety hints advertised to clients."""

    readOnlyHint: bool = True
    destructiveHint: bool = False


class MCPToolDescriptor(BaseModel):
    """Tool metadata returned by `tools/list`."""

    name: str
    description: str
    inputSchema: dict[str, Any]
    annotations: MCPToolAnnotations = Field(default_factory=MCPToolAnnotations)


class MCPResourceDescriptor(BaseModel):
    """Resource metadata returned by `resources/list`."""

    uri: str
    name: str
    description: str
    mimeType: str = "text/plain"
