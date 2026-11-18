"""Pydantic v2 JSON Schema Auto-Registry & @tool Decorator."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from react_mcp_runtime.schema import RiskLevel


class ToolExecutionError(RuntimeError):
    """Raised when a tool encounters an unrecoverable runtime failure."""


class ToolSecurityBlockedError(PermissionError):
    """Raised when SQL Read-Only Guard or HITL Interceptor blocks a tool call."""

    def __init__(self, message: str, *, reason_code: str = "policy_denied") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Produce concise OpenAI function parameter JSON Schema without root title clutter."""
    cleaned = dict(schema)
    cleaned.pop("title", None)
    if "properties" in cleaned:
        props: dict[str, Any] = {}
        for key, prop in cleaned["properties"].items():
            p = dict(prop)
            p.pop("title", None)
            props[key] = p
        cleaned["properties"] = props
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


def build_model_from_callable(func: Callable[..., Any], model_name: str) -> type[BaseModel]:
    """Extract parameter types, defaults, and Field metadata into a strict Pydantic v2 model."""
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def build_model_from_json_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Dynamically construct a strict Pydantic v2 BaseModel from an MCP JSON Schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    type_map: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict[str, Any],
    }

    for prop_name, prop_schema in properties.items():
        if "enum" in prop_schema and prop_schema["enum"]:
            enum_vals = tuple(prop_schema["enum"])
            py_type: Any = Literal[enum_vals]  # type: ignore[valid-type]
        elif prop_schema.get("type") == "array":
            item_type_name = prop_schema.get("items", {}).get("type", "string")
            item_py = type_map.get(item_type_name, Any)
            py_type = list[item_py]  # type: ignore[valid-type]
        else:
            raw_type = prop_schema.get("type", "string")
            if isinstance(raw_type, list):
                non_null = [t for t in raw_type if t != "null"]
                base_t = type_map.get(non_null[0], Any) if non_null else Any
                py_type = base_t | None
            else:
                py_type = type_map.get(raw_type, Any)

        field_kwargs: dict[str, Any] = {}
        if "description" in prop_schema:
            field_kwargs["description"] = prop_schema["description"]
        if "minimum" in prop_schema:
            field_kwargs["ge"] = prop_schema["minimum"]
        if "maximum" in prop_schema:
            field_kwargs["le"] = prop_schema["maximum"]
        if "minLength" in prop_schema:
            field_kwargs["min_length"] = prop_schema["minLength"]

        is_nullable = isinstance(prop_schema.get("type"), list) and "null" in prop_schema["type"]
        if prop_name in required and "default" not in prop_schema:
            fields[prop_name] = (py_type, Field(..., **field_kwargs))
        else:
            default_val = prop_schema.get("default", None)
            opt_type = py_type | None if (default_val is None and not is_nullable) else py_type
            fields[prop_name] = (opt_type, Field(default=default_val, **field_kwargs))

    return create_model(
        f"{name}_MCPArgs",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class RegisteredTool:
    """Encapsulates a validated callable tool with OpenAI & MCP compatible schemas."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        args_model: type[BaseModel],
        risk_level: RiskLevel = "read",
        raw_schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_model = args_model
        self.risk_level = risk_level
        self._is_async = inspect.iscoroutinefunction(func)
        self.parameters_schema = (
            _clean_json_schema(raw_schema)
            if raw_schema is not None
            else _clean_json_schema(args_model.model_json_schema())
        )

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_arguments(self, raw_args: dict[str, Any]) -> dict[str, Any]:
        validated = self.args_model.model_validate(raw_args)
        return {
            field_name: getattr(validated, field_name)
            for field_name in self.args_model.model_fields
        }

    async def invoke(self, raw_args: dict[str, Any]) -> str:
        clean_kwargs = self.validate_arguments(raw_args)
        result = await self.func(**clean_kwargs) if self._is_async else self.func(**clean_kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return json.dumps(result, ensure_ascii=False, default=str)


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    risk_level: RiskLevel = "read",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator attaching a RegisteredTool specification (`__tool_spec__`) to a function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        doc = inspect.getdoc(func) or f"Execute {tool_name}."
        tool_desc = description or doc.splitlines()[0].strip()
        args_model = build_model_from_callable(func, f"{tool_name}_Args")
        spec = RegisteredTool(
            name=tool_name,
            description=tool_desc,
            func=func,
            args_model=args_model,
            risk_level=risk_level,
        )
        func.__tool_spec__ = spec  # type: ignore[attr-defined]
        return func

    return decorator


class ToolRegistry:
    """In-memory registry mapping tool names to Pydantic-validated RegisteredTool handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        item: Callable[..., Any] | RegisteredTool,
        *,
        name: str | None = None,
        description: str | None = None,
        risk_level: RiskLevel = "read",
    ) -> RegisteredTool:
        if isinstance(item, RegisteredTool):
            spec = item
        elif hasattr(item, "__tool_spec__"):
            spec = item.__tool_spec__  # type: ignore[attr-defined]
        else:
            decorated = tool(name=name, description=description, risk_level=risk_level)(item)
            spec = decorated.__tool_spec__  # type: ignore[attr-defined]
        self._tools[spec.name] = spec
        return spec

    def register_json_schema_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
        risk_level: RiskLevel = "read",
    ) -> RegisteredTool:
        args_model = build_model_from_json_schema(name, input_schema)
        spec = RegisteredTool(
            name=name,
            description=description,
            func=handler,
            args_model=args_model,
            risk_level=risk_level,
            raw_schema=input_schema,
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


__all__ = [
    "RegisteredTool",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolSecurityBlockedError",
    "ValidationError",
    "build_model_from_callable",
    "build_model_from_json_schema",
    "tool",
]
