import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.orchestration.models import ExecutionPlan, PlanLimits, PlanStatus, StepStatus, StepType


class MissingStepOutput(ValueError):
    pass


@dataclass
class StepExecutionResult:
    output: Any = None
    paused_for_approval: bool = False
    executed_tool: bool = False
    error: Exception | None = None


@dataclass
class PlanAdvanceResult:
    paused_for_approval: bool
    elapsed_seconds: float
    tool_calls_used: int


def _read_reference(reference: str, outputs: dict[str, Any]) -> Any:
    parts = reference.split(".")
    step_id = parts[0]
    if step_id not in outputs:
        raise MissingStepOutput(f"Output for step '{step_id}' is not available.")
    value = outputs[step_id]
    for key in parts[2:]:
        try:
            if isinstance(value, dict):
                value = value[key]
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                raise KeyError(key)
        except (KeyError, IndexError) as exc:
            raise MissingStepOutput(f"Output reference '{reference}' cannot be resolved.") from exc
    return value


def resolve_references(value: Any, outputs: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return _read_reference(value["$ref"], outputs)
        return {key: resolve_references(child, outputs) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_references(child, outputs) for child in value]
    if isinstance(value, str):
        import re
        full = re.fullmatch(r"\$\{([a-zA-Z0-9_-]+\.output(?:\.[a-zA-Z0-9_-]+)*)\}", value)
        if full:
            return _read_reference(full.group(1), outputs)
        return re.sub(
            r"\$\{([a-zA-Z0-9_-]+\.output(?:\.[a-zA-Z0-9_-]+)*)\}",
            lambda match: str(_read_reference(match.group(1), outputs)), value,
        )
    return value


class PlanExecutionEngine:
    """Sequential DAG scheduler; independent ready steps stay eligible after failures."""

    def __init__(self, limits: PlanLimits):
        self.limits = limits

    async def advance(
        self,
        plan: ExecutionPlan,
        outputs: dict[str, Any],
        execute_step: Callable[[Any, dict[str, Any]], Any],
        record: Callable[[str, dict[str, Any]], None],
        *,
        active_elapsed_seconds: float = 0.0,
        tool_calls_used: int = 0,
    ) -> PlanAdvanceResult:
        started = time.monotonic()
        steps = {step.step_id: step for step in plan.steps}
        plan.status = PlanStatus.RUNNING
        record("plan_execution_started", {"plan_id": plan.plan_id})

        for step_id in plan.execution_order:
            step = steps[step_id]
            if step.status == StepStatus.COMPLETED:
                continue
            failed_dependencies = [
                dependency for dependency in step.depends_on
                if steps[dependency].status in (StepStatus.FAILED, StepStatus.BLOCKED, StepStatus.SKIPPED)
            ]
            if failed_dependencies:
                step.status = StepStatus.BLOCKED
                step.error = "A required dependency did not complete."
                record("step_blocked", {"plan_id": plan.plan_id, "step_id": step_id,
                                         "failed_dependencies": failed_dependencies})
                continue

            pending_dependencies = [dependency for dependency in step.depends_on
                                     if steps[dependency].status != StepStatus.COMPLETED]
            if pending_dependencies:
                step.status = StepStatus.BLOCKED
                step.error = "A dependency was not ready in execution order."
                record("step_blocked", {"plan_id": plan.plan_id, "step_id": step_id,
                                         "pending_dependencies": pending_dependencies})
                continue

            if active_elapsed_seconds + time.monotonic() - started >= self.limits.max_duration_seconds:
                step.status = StepStatus.FAILED
                step.error = "Plan execution duration limit reached."
                record("step_failed", {"plan_id": plan.plan_id, "step_id": step_id,
                                        "reason": "execution_duration_limit"})
                for remaining_id in plan.execution_order[plan.execution_order.index(step_id) + 1:]:
                    remaining = steps[remaining_id]
                    if remaining.status == StepStatus.PENDING:
                        remaining.status = StepStatus.SKIPPED
                        record("step_skipped", {"plan_id": plan.plan_id, "step_id": remaining_id,
                                                 "reason": "execution_limit"})
                break

            if step.step_type == StepType.TOOL and tool_calls_used >= self.limits.max_tool_calls:
                step.status = StepStatus.FAILED
                step.error = "Plan tool-call limit reached."
                record("step_failed", {"plan_id": plan.plan_id, "step_id": step_id,
                                        "reason": "tool_call_limit"})
                continue

            step.status = StepStatus.READY
            record("step_started", {
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "step_type": step.step_type.value,
                "dependencies": step.depends_on,
                "satisfied_dependencies": list(step.depends_on),
                "input_refs": step.input_refs,
            })
            step.status = StepStatus.RUNNING
            try:
                resolved_inputs = {reference: _read_reference(reference, outputs)
                                  for reference in step.input_refs}
                resolved_arguments = resolve_references(step.arguments, outputs)
                result = await execute_step(step, {"inputs": resolved_inputs,
                                                    "arguments": resolved_arguments})
                if result.paused_for_approval:
                    step.status = StepStatus.AWAITING_APPROVAL
                    plan.status = PlanStatus.AWAITING_APPROVAL
                    record("plan_paused_for_approval", {
                        "plan_id": plan.plan_id, "step_id": step_id,
                        "tool_call_id": step.tool_call_id,
                    })
                    return PlanAdvanceResult(True, active_elapsed_seconds + time.monotonic() - started,
                                             tool_calls_used)
                if result.executed_tool:
                    tool_calls_used += 1
                if result.error is not None:
                    raise result.error
                encoded = json.dumps(result.output, default=str).encode("utf-8")
                if len(encoded) > self.limits.max_step_output_bytes:
                    raise ValueError("Step output exceeds the configured size limit.")
                step.output = result.output
                step.output_refs = [f"{step_id}.output"]
                outputs[step_id] = result.output
                step.status = StepStatus.COMPLETED
                record("step_completed", {"plan_id": plan.plan_id, "step_id": step_id,
                                           "output_bytes": len(encoded),
                                           "available_output_refs": [f"{step_id}.output"]})
            except Exception as exc:
                step.status = StepStatus.FAILED
                safe_reason = (str(exc) if isinstance(exc, (MissingStepOutput, PermissionError))
                               else "Execution step failed.")
                step.error = f"{type(exc).__name__}: {safe_reason}"
                record("step_failed", {"plan_id": plan.plan_id, "step_id": step_id,
                                        "error_type": type(exc).__name__, "reason": safe_reason[:300]})

        active_elapsed_seconds += time.monotonic() - started
        if all(step.status == StepStatus.COMPLETED for step in plan.steps):
            plan.status = PlanStatus.COMPLETED
            return PlanAdvanceResult(False, active_elapsed_seconds, tool_calls_used)
        plan.status = PlanStatus.FAILED
        if plan.error is None:
            plan.error = "One or more execution steps failed or were blocked."
        return PlanAdvanceResult(False, active_elapsed_seconds, tool_calls_used)
