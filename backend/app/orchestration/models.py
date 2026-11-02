from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StepType(StrEnum):
    TOOL = "tool_execution"
    REASONING = "llm_reasoning"
    FINAL = "final_response"


class StepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class PlanStatus(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(min_length=1, max_length=500)
    step_type: StepType
    depends_on: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    input_summary: str | None = None
    output: Any | None = None
    output_refs: list[str] = Field(default_factory=list)
    error: str | None = None
    tool_call_id: str | None = None
    approval_id: str | None = None
    tool_call_logged: bool = False


class PlanStepProposal(BaseModel):
    """Untrusted planner-controlled fields; execution state is runtime-owned."""

    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(min_length=1, max_length=500)
    step_type: StepType
    depends_on: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlanProposal(BaseModel):
    """Only fields the planning LLM is permitted to propose."""

    model_config = ConfigDict(extra="forbid")
    steps: list[PlanStepProposal] = Field(min_length=1)


class ExecutionPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    original_request: str
    steps: list[PlanStep]
    execution_order: list[str] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.PROPOSED
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    error: str | None = None


class PlanLimits(BaseModel):
    max_steps: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=8, ge=0)
    max_duration_seconds: float = Field(default=60.0, gt=0)
    max_step_output_bytes: int = Field(default=32_000, ge=1)
    tool_retry_limit: int = Field(default=1, ge=0)

