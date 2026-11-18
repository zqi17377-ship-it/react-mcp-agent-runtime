"""Unit tests for Pydantic v2 JSON Schema extraction, @tool decorator, and dynamic MCP validation."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from react_mcp_runtime.tools import ToolRegistry, build_model_from_json_schema, tool


@tool(description="Compute rolling latency threshold check.")
def check_slo(
    service: str,
    window_minutes: int = Field(default=5, ge=1, le=60),
    severity: Literal["warn", "critical"] = "critical",
) -> dict[str, object]:
    return {"service": service, "window_minutes": window_minutes, "severity": severity}


@tool(name="async_echo_metric", risk_level="write")
async def _async_metric_writer(metric_name: str, value: float) -> str:
    return f"{metric_name}={value:.2f}"


def test_tool_decorator_extracts_pydantic_schema() -> None:
    registry = ToolRegistry()
    spec = registry.register(check_slo)

    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "check_slo"
    assert fn["description"] == "Compute rolling latency threshold check."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "service" in params["required"]
    assert params["properties"]["window_minutes"]["minimum"] == 1
    assert params["properties"]["window_minutes"]["maximum"] == 60
    assert set(params["properties"]["severity"]["enum"]) == {"warn", "critical"}


def test_tool_rejects_unknown_hallucinated_extra_fields() -> None:
    registry = ToolRegistry()
    spec = registry.register(check_slo)

    with pytest.raises(ValidationError) as exc_info:
        spec.validate_arguments({"service": "checkout-api", "hallucinated_flag": True})
    assert any("extra" in err["type"] for err in exc_info.value.errors())

    with pytest.raises(ValidationError):
        spec.validate_arguments({"service": "checkout-api", "window_minutes": 999})


async def test_async_and_sync_tool_invocation() -> None:
    registry = ToolRegistry()
    registry.register(check_slo)
    registry.register(_async_metric_writer)

    sync_spec = registry.get("check_slo")
    async_spec = registry.get("async_echo_metric")
    assert sync_spec is not None and sync_spec.risk_level == "read"
    assert async_spec is not None and async_spec.risk_level == "write"

    res_sync = await sync_spec.invoke({"service": "catalog-api", "window_minutes": 15})
    assert '"service": "catalog-api"' in res_sync
    assert '"window_minutes": 15' in res_sync

    res_async = await async_spec.invoke({"metric_name": "p99_ms", "value": 84.356})
    assert res_async == "p99_ms=84.36"


def test_dynamic_mcp_json_schema_model_builder() -> None:
    mcp_schema = {
        "type": "object",
        "properties": {
            "service": {"type": ["string", "null"], "default": None},
            "min_level": {"type": "string", "enum": ["INFO", "WARN", "ERROR"], "default": "ERROR"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "additionalProperties": False,
    }
    model = build_model_from_json_schema("analyze_logs", mcp_schema)
    valid = model.model_validate({"service": "checkout-api", "min_level": "WARN", "top_k": 3})
    assert valid.service == "checkout-api"  # type: ignore[attr-defined]
    assert valid.top_k == 3  # type: ignore[attr-defined]

    with pytest.raises(ValidationError):
        model.model_validate({"min_level": "NOT_A_LEVEL"})

    with pytest.raises(ValidationError):
        model.model_validate({"top_k": 0})
