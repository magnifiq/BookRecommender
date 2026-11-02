from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScriptedToolCall(BaseModel):
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None


class ScriptedResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)
    error: Literal["connection", "timeout", "http_429", "http_500", "http_400"] | None = None


class ToolResultAssertion(BaseModel):
    tool_name: str
    result_path: str
    expected_value: Any


class EvaluationScenario(BaseModel):
    scenario_id: str
    category: str
    user_input: str
    conversation_turns: list[str] = Field(default_factory=list)
    plan_proposal: dict[str, Any] | None = None
    planning_error: bool = False
    expected_plan_status: Literal[
        "not_planned", "validated", "running", "awaiting_approval",
        "completed", "failed", "rejected",
    ] | None = None
    expected_step_status: list[str] | None = None
    planning_expected: bool | None = None
    expects_recovery: bool = False
    plan_limit_overrides: dict[str, int | float] = Field(default_factory=dict)
    expected_behavior: str
    expected_run_status: Literal["completed", "failed", "awaiting_approval"]
    expected_tool_calls: list[str] | None = None
    expected_guardrail: Literal[
        "input_blocked", "tool_blocked", "output_rejected", "none"
    ] | None = None
    expected_permission: Literal["auto", "denied", "requires_approval"] | None = None
    approval_decision: Literal["approved", "rejected"] | None = None
    expected_tool_status: list[str] | None = None
    expected_retry_count: int | None = None
    tool_result_assertions: list[ToolResultAssertion] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    scripted_responses: list[ScriptedResponse] = Field(default_factory=list)
    tool_fixtures: dict[str, Literal["success", "failure", "timeout", "connection_once"]] = Field(default_factory=dict)
    provider_books: list[dict[str, Any]] = Field(default_factory=list)
    provider_unavailable: bool = False


class EvaluationCheck(BaseModel):
    check_id: str
    passed: bool
    expected: str
    actual: str
    explanation: str | None = None


class QualitativeEvaluation(BaseModel):
    evaluator_model: str
    rubric_version: str
    scores: dict[str, float]
    notes: str | None = None


class EvaluationCaseResult(BaseModel):
    scenario_id: str
    category: str
    passed: bool
    expected_behavior: str
    actual_behavior: str
    checks: list[EvaluationCheck] = Field(default_factory=list)
    trace_id: str | None = None
    failure_explanation: str | None = None
    qualitative_evaluation: QualitativeEvaluation | None = None
    qualitative_error: str | None = None


class EvaluationMetrics(BaseModel):
    scenario_count: int = 0
    task_completion_rate: float = 0.0
    expected_tool_selection_accuracy: float = 0.0
    unexpected_tool_invocation_rate: float = 0.0
    correct_rejection_handling_rate: float = 0.0
    successful_run_rate: float = 0.0
    tool_execution_failure_rate: float = 0.0
    retry_success_rate: float = 0.0
    timeout_rate: float = 0.0
    guardrail_enforcement_rate: float = 0.0
    permission_enforcement_rate: float = 0.0
    denied_operation_execution_rate: float = 0.0
    approval_bypass_rate: float = 0.0
    average_run_duration_ms: float = 0.0
    average_tool_execution_duration_ms: float = 0.0
    average_llm_requests_per_run: float = 0.0
    average_tool_calls_per_run: float = 0.0
    plan_validity_rate: float = 0.0
    step_completion_rate: float = 0.0
    dependency_enforcement_rate: float = 0.0
    unnecessary_planning_rate: float = 0.0
    duplicate_execution_rate: float = 0.0
    successful_recovery_rate: float = 0.0
    average_steps_per_completed_task: float = 0.0


class EvaluationCategorySummary(BaseModel):
    category: str
    scenario_count: int
    passed_count: int
    failed_count: int
    pass_rate: float


class EvaluationRun(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    status: Literal["running", "completed", "failed"] = "running"
    execution_mode: Literal["scripted", "live"] = "scripted"
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    duration_ms: float | None = None
    case_results: list[EvaluationCaseResult] = Field(default_factory=list)
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    categories: list[EvaluationCategorySummary] = Field(default_factory=list)
    error: str | None = None

    def finish(self) -> None:
        self.finished_at = utc_now()
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000
        self.status = "completed"

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
