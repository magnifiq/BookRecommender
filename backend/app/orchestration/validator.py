import re
from collections import defaultdict, deque
from uuid import uuid4

from app.orchestration.models import (
    ExecutionPlan, PlanProposal, PlanStatus, PlanStep, PlanStepProposal, StepType,
)
from app.tools.registry import ToolRegistry


class InvalidPlan(ValueError):
    pass


_REFERENCE = re.compile(r"\$\{([a-zA-Z0-9_-]+\.output(?:\.[a-zA-Z0-9_-]+)*)\}")


def _walk_refs(value):
    if isinstance(value, dict):
        if set(value) == {"$ref"} and isinstance(value["$ref"], str):
            yield value["$ref"]
        else:
            for child in value.values():
                yield from _walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_refs(child)
    elif isinstance(value, str):
        yield from _REFERENCE.findall(value)


class PlanValidator:
    def __init__(self, registry: ToolRegistry, max_steps: int = 12):
        self.registry = registry
        self.max_steps = max_steps

    def validate(self, proposal: PlanProposal, run_id: str, original_request: str) -> ExecutionPlan:
        if len(proposal.steps) > self.max_steps:
            raise InvalidPlan(f"Plan exceeds maximum step count ({self.max_steps}).")
        steps = proposal.steps
        by_id = {step.step_id: step for step in steps}
        if len(by_id) != len(steps):
            raise InvalidPlan("Plan step IDs must be unique.")
        final_steps = [step for step in steps if step.step_type == StepType.FINAL]
        if len(final_steps) != 1 or steps[-1].step_type != StepType.FINAL:
            raise InvalidPlan("Plan must contain exactly one final response step, last in the plan.")

        parents: dict[str, set[str]] = defaultdict(set)
        children: dict[str, set[str]] = defaultdict(set)
        indegree = {step.step_id: 0 for step in steps}
        for step in steps:
            if len(set(step.depends_on)) != len(step.depends_on):
                raise InvalidPlan(f"Step '{step.step_id}' has duplicate dependencies.")
            for dependency in step.depends_on:
                if dependency not in by_id or dependency == step.step_id:
                    raise InvalidPlan(f"Step '{step.step_id}' has an invalid dependency '{dependency}'.")
                parents[step.step_id].add(dependency)
                children[dependency].add(step.step_id)
                indegree[step.step_id] += 1
            if step.step_type == StepType.TOOL:
                if not step.tool_name:
                    raise InvalidPlan(f"Tool step '{step.step_id}' requires a tool name.")
                try:
                    self.registry.get(step.tool_name)
                except KeyError as exc:
                    raise InvalidPlan(f"Plan references unknown tool '{step.tool_name}'.") from exc
            elif step.tool_name is not None or step.arguments:
                raise InvalidPlan(f"Non-tool step '{step.step_id}' cannot define a tool or arguments.")

            actual_refs = set(_walk_refs(step.arguments))
            declared_refs = set(step.input_refs)
            if actual_refs - declared_refs:
                raise InvalidPlan(f"Step '{step.step_id}' contains undeclared output references.")
            for reference in declared_refs:
                parts = reference.split(".")
                if (len(parts) < 2 or parts[1] != "output" or parts[0] not in by_id
                        or not re.fullmatch(r"[a-zA-Z0-9_-]+\.output(?:\.[a-zA-Z0-9_-]+)*", reference)):
                    raise InvalidPlan(f"Step '{step.step_id}' has invalid output reference '{reference}'.")
                if parts[0] not in parents[step.step_id]:
                    raise InvalidPlan(f"Step '{step.step_id}' must depend directly on reference source '{parts[0]}'.")

        ready = deque(step.step_id for step in steps if indegree[step.step_id] == 0)
        order = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(order) != len(steps):
            raise InvalidPlan("Plan dependencies contain a cycle.")

        final_id = final_steps[0].step_id

        runtime_steps = []
        for proposal_step in steps:
            values = proposal_step.model_dump()
            if proposal_step.step_type == StepType.TOOL:
                values["tool_call_id"] = f"plan-{uuid4()}"
            runtime_steps.append(PlanStep(**values))
        return ExecutionPlan(
            run_id=run_id,
            original_request=original_request,
            steps=runtime_steps,
            execution_order=[step_id for step_id in order if step_id != final_id] + [final_id],
            status=PlanStatus.VALIDATED,
        )
