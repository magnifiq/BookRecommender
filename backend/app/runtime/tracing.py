
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceEvent(BaseModel):
    """A single event in an agent execution."""

    event_type: str
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(BaseModel):
    """Records the execution of one tool call."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]

    status: Literal["running", "awaiting_approval", "completed", "failed"] = "running"

    result: Any | None = None
    error: str | None = None
    approval_id: str | None = None

    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    duration_ms: float | None = None


class AgentRun(BaseModel):
    """Represents one complete agent execution."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))

    user_message: str

    status: Literal["running", "awaiting_approval", "completed", "failed"] = "running"

    events: list[TraceEvent] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)

    final_response: str | None = None
    error: str | None = None

    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    duration_ms: float | None = None
