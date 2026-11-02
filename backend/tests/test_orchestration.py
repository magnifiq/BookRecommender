from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.orchestration.engine import PlanExecutionEngine
from app.orchestration.models import ExecutionPlan, PlanLimits, PlanProposal, PlanStatus, PlanStep, StepStatus, StepType
from app.orchestration.planner import should_plan
from app.orchestration.validator import InvalidPlan, PlanValidator
from app.runtime.agent import AgentRuntime
from app.runtime.approvals import ApprovalStore
from app.runtime.trace_collector import TraceCollector
from app.tools.base import PermissionPolicy, ToolDefinition
from app.tools.registry import ToolRegistry


class QueryArgs(BaseModel):
    query: str


class BookListArgs(BaseModel):
    books: list[dict]


class ApprovalArgs(BaseModel):
    note: str


class StaticPlanner:
    def __init__(self, proposal=None, error=None):
        self.proposal = proposal
        self.error = error
        self.calls = 0

    async def propose(self, _request, _registry, _invoke, context_messages=None):
        self.calls += 1
        if self.error:
            raise self.error
        return PlanProposal.model_validate(self.proposal)


def assistant(content=None, tool_calls=()):
    dumped = [{"id": call.id, "type": "function", "function": {
        "name": call.function.name, "arguments": call.function.arguments,
    }} for call in tool_calls]
    message = SimpleNamespace(
        content=content, tool_calls=list(tool_calls),
        model_dump=lambda **_: {"role": "assistant", "content": content,
                                **({"tool_calls": dumped} if dumped else {})},
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="tool_calls" if dumped else "stop"
    )])


def make_runtime(registry, planner, llm_contents=(), **kwargs):
    responses = list(llm_contents)
    prompts = []

    async def create(**params):
        prompts.append(params)
        if not responses:
            raise AssertionError("Unexpected LLM request")
        return assistant(content=responses.pop(0))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    traces = TraceCollector()
    runtime = AgentRuntime(client, "mock", registry, traces, plan_planner=planner, **kwargs)
    return runtime, traces, prompts


def simple_plan(tool_name="search", args=None, policy=None):
    return {"steps": [
        {"step_id": "lookup", "description": "Look up books", "step_type": "tool_execution",
         "tool_name": tool_name, "arguments": args or {"query": "AI"}},
        {"step_id": "answer", "description": "Answer the user", "step_type": "final_response",
         "depends_on": ["lookup"], "input_refs": ["lookup.output"]},
    ]}


def registry_for(handler, policy=PermissionPolicy.AUTO, tool_name="search", args=QueryArgs, retry_safe=False):
    registry = ToolRegistry()
    registry.register(ToolDefinition(tool_name, "Synthetic tool", args, handler, policy, retry_safe))
    return registry


@pytest.mark.asyncio
async def test_simple_prompt_does_not_trigger_planning():
    planner = StaticPlanner(proposal=simple_plan())
    runtime, _, _ = make_runtime(ToolRegistry(), planner, ["A simple answer."])
    run = await runtime.run("Recommend a fantasy book")
    assert run.final_response == "A simple answer."
    assert planner.calls == 0
    assert run.execution_plan is None


@pytest.mark.asyncio
async def test_complex_book_plan_runs_dependencies_and_passes_only_declared_outputs():
    calls = []

    async def search(args):
        calls.append(args.query)
        return {"books": [{"title": "Dune", "year": 1965}]}

    registry = registry_for(search)
    proposal = {"steps": [
        {"step_id": "search", "description": "Find AI science-fiction books", "step_type": "tool_execution",
         "tool_name": "search", "arguments": {"query": "AI science fiction"}},
        {"step_id": "compare", "description": "Compare available book years", "step_type": "llm_reasoning",
         "depends_on": ["search"], "input_refs": ["search.output.books"]},
        {"step_id": "recommend", "description": "Recommend the strongest choice", "step_type": "final_response",
         "depends_on": ["compare"], "input_refs": ["compare.output"]},
    ]}
    planner = StaticPlanner(proposal)
    runtime, traces, prompts = make_runtime(registry, planner, ["Dune is the strongest match.", "Choose Dune, first published in 1965."])
    run = await runtime.run("Find AI science fiction, compare publication years, then recommend the best one.")
    assert run.status == "completed"
    assert run.execution_plan.status == PlanStatus.COMPLETED
    assert [step.status for step in run.execution_plan.steps] == [StepStatus.COMPLETED] * 3
    assert calls == ["AI science fiction"]
    reasoning_prompt = prompts[0]["messages"][1]["content"]
    assert "Dune" in reasoning_prompt and "available_inputs" in reasoning_prompt
    assert run.final_response == "Choose Dune, first published in 1965."
    event_types = [event.event_type for event in traces.get_run(run.run_id).events]
    assert "plan_created" in event_types and "plan_validated" in event_types
    assert event_types.index("step_completed") < event_types.index("plan_completed")


def test_plan_validator_rejects_unknown_tool_cycles_and_undeclared_references():
    registry = registry_for(lambda _: None)
    validator = PlanValidator(registry)

    unknown = simple_plan(tool_name="not-registered")
    with pytest.raises(InvalidPlan, match="unknown tool"):
        validator.validate(PlanProposal.model_validate(unknown), "r1", "request")

    cycle = {"steps": [
        {"step_id": "a", "description": "A", "step_type": "llm_reasoning", "depends_on": ["b"]},
        {"step_id": "b", "description": "B", "step_type": "llm_reasoning", "depends_on": ["a"]},
        {"step_id": "final", "description": "Answer", "step_type": "final_response"},
    ]}
    with pytest.raises(InvalidPlan, match="cycle"):
        validator.validate(PlanProposal.model_validate(cycle), "r1", "request")

    missing_ref = {"steps": [
        {"step_id": "a", "description": "A", "step_type": "tool_execution", "tool_name": "search",
         "arguments": {"query": {"$ref": "other.output.query"}}},
        {"step_id": "final", "description": "Answer", "step_type": "final_response", "depends_on": ["a"]},
    ]}
    with pytest.raises(InvalidPlan):
        validator.validate(PlanProposal.model_validate(missing_ref), "r1", "request")

    with pytest.raises(Exception):
        PlanProposal.model_validate({"steps": [], "permission_policy": "auto"})


@pytest.mark.asyncio
async def test_unknown_tool_plan_is_rejected_then_uses_existing_agent_loop():
    planner = StaticPlanner(simple_plan(tool_name="missing"))
    runtime, traces, _ = make_runtime(ToolRegistry(), planner, ["Fallback answer."])
    run = await runtime.run("Find three books, compare them, then recommend one.")
    assert run.status == "completed"
    assert run.execution_plan.status == PlanStatus.REJECTED
    assert "plan_rejected" in [event.event_type for event in run.events]
    assert run.final_response == "Fallback answer."
    assert runtime._plan_states == {}


@pytest.mark.asyncio
async def test_approval_pauses_and_resumes_plan_without_repeating_completed_steps():
    executed = []

    async def search(args):
        executed.append("search")
        return {"books": [args.query]}

    async def approved(args):
        executed.append(args.note)
        return {"approved": args.note}

    registry = ToolRegistry()
    registry.register(ToolDefinition("search", "Search", QueryArgs, search))
    registry.register(ToolDefinition("approval", "Approval demo", ApprovalArgs, approved,
                                     PermissionPolicy.REQUIRES_APPROVAL))
    proposal = {"steps": [
        {"step_id": "search", "description": "Search first", "step_type": "tool_execution",
         "tool_name": "search", "arguments": {"query": "Dune"}},
        {"step_id": "confirm", "description": "Run approved lookup", "step_type": "tool_execution",
         "depends_on": ["search"], "tool_name": "approval", "arguments": {"note": "Dune details"}},
        {"step_id": "final", "description": "Respond", "step_type": "final_response",
         "depends_on": ["confirm"], "input_refs": ["confirm.output"]},
    ]}
    runtime, traces, _ = make_runtime(registry, StaticPlanner(proposal), ["The lookup is complete."])
    pending = await runtime.run("Find Dune, compare its details, then recommend the best one.")
    assert pending.status == "awaiting_approval"
    assert pending.execution_plan.status == PlanStatus.AWAITING_APPROVAL
    assert executed == ["search"]
    request = runtime.approval_store.list_pending()[0]
    call_id = request.tool_call_id
    complete = await runtime.resolve_approval(request.approval_id, "approved")
    assert complete.status == "completed"
    assert executed == ["search", "Dune details"]
    assert complete.execution_plan.steps[0].status == StepStatus.COMPLETED
    assert complete.execution_plan.steps[1].status == StepStatus.COMPLETED
    assert [tool.tool_call_id for tool in complete.tool_calls if tool.tool_name == "approval"] == [call_id]
    assert "plan_resumed" in [event.event_type for event in traces.get_run(complete.run_id).events]
    assert not runtime._plan_states


@pytest.mark.asyncio
async def test_denied_plan_tool_never_executes_and_blocks_dependent_final_step():
    async def forbidden(_args):
        pytest.fail("denied handler must not execute")

    registry = registry_for(forbidden, policy=PermissionPolicy.DENIED)
    runtime, traces, _ = make_runtime(registry, StaticPlanner(simple_plan()), [])
    run = await runtime.run("Find three books, compare them, then recommend one.")
    assert run.status == "failed"
    assert run.execution_plan.steps[0].status == StepStatus.FAILED
    assert run.execution_plan.steps[1].status == StepStatus.BLOCKED
    events = [event.event_type for event in traces.get_run(run.run_id).events]
    assert "tool_execution_denied" in events and "step_blocked" in events
    assert "tool_call_started" not in events


@pytest.mark.asyncio
async def test_failed_dependency_blocks_child_but_independent_step_is_allowed_to_run():
    calls = []

    async def search(args):
        calls.append(args.query)
        if args.query == "failure":
            raise RuntimeError("synthetic failure")
        return {"query": args.query}

    registry = registry_for(search)
    proposal = {"steps": [
        {"step_id": "failed", "description": "Failing lookup", "step_type": "tool_execution",
         "tool_name": "search", "arguments": {"query": "failure"}},
        {"step_id": "independent", "description": "Independent lookup", "step_type": "tool_execution",
         "tool_name": "search", "arguments": {"query": "independent"}},
        {"step_id": "dependent", "description": "Needs failed lookup", "step_type": "llm_reasoning",
         "depends_on": ["failed"], "input_refs": ["failed.output"]},
        {"step_id": "final", "description": "Respond", "step_type": "final_response",
         "depends_on": ["dependent"]},
    ]}
    runtime, _, _ = make_runtime(registry, StaticPlanner(proposal), [])
    run = await runtime.run("Research the books, compare them, then recommend the best one.")
    assert run.status == "failed"
    statuses = {step.step_id: step.status for step in run.execution_plan.steps}
    assert statuses["failed"] == StepStatus.FAILED
    assert statuses["independent"] == StepStatus.COMPLETED
    assert statuses["dependent"] == StepStatus.BLOCKED
    assert calls == ["failure", "independent"]


@pytest.mark.asyncio
async def test_retry_safe_tool_retries_transient_failure_once():
    attempts = []

    async def flaky(args):
        attempts.append(args.query)
        if len(attempts) == 1:
            raise ConnectionError("transient")
        return {"books": ["Dune"]}

    registry = registry_for(flaky, retry_safe=True)
    runtime, traces, _ = make_runtime(registry, StaticPlanner(simple_plan()), ["Dune is a match."])
    run = await runtime.run("Find books and compare them, then recommend one.")
    assert run.status == "completed"
    assert attempts == ["AI", "AI"]
    assert any(event.event_type == "tool_retry_scheduled" for event in traces.get_run(run.run_id).events)


@pytest.mark.asyncio
async def test_plan_limits_stop_before_unbounded_execution():
    calls = []

    async def search(args):
        calls.append(args.query)
        return {"books": []}

    registry = registry_for(search)
    runtime, traces, _ = make_runtime(
        registry, StaticPlanner(simple_plan()), [],
        plan_limits=PlanLimits(max_tool_calls=0),
    )
    run = await runtime.run("Find three books, compare them, then recommend one.")
    assert run.status == "failed"
    assert not calls
    assert any(event.details.get("reason") == "tool_call_limit"
               for event in traces.get_run(run.run_id).events if event.event_type == "step_failed")


@pytest.mark.asyncio
async def test_duration_limit_prevents_steps_after_budget_is_exhausted():
    plan = ExecutionPlan(
        run_id="run", original_request="request", status=PlanStatus.VALIDATED,
        steps=[PlanStep(step_id="final", description="Answer", step_type=StepType.FINAL)],
        execution_order=["final"],
    )
    executed = []
    events = []

    async def execute(*_args):
        executed.append(True)

    result = await PlanExecutionEngine(PlanLimits(max_duration_seconds=1)).advance(
        plan, {}, execute, lambda name, details: events.append((name, details)),
        active_elapsed_seconds=1,
    )
    assert not executed
    assert plan.status == PlanStatus.FAILED
    assert any(event == "step_failed" and details["reason"] == "execution_duration_limit"
               for event, details in events)


@pytest.mark.asyncio
async def test_approval_expiry_fails_plan_without_executing_tool():
    async def lookup(_args):
        pytest.fail("expired approval must not execute")

    registry = registry_for(lookup, policy=PermissionPolicy.REQUIRES_APPROVAL)
    runtime, _, _ = make_runtime(registry, StaticPlanner(simple_plan()), [])
    pending = await runtime.run("Find books, compare the results, then recommend one.")
    approval = runtime.approval_store.list_pending()[0]
    approval.expires_at = approval.created_at
    from app.runtime.approvals import ApprovalConflict
    with pytest.raises(ApprovalConflict, match="expired"):
        await runtime.resolve_approval(approval.approval_id, "approved")
    assert pending.execution_plan.status == PlanStatus.FAILED
    assert pending.tool_calls[0].status == "failed"


def test_planning_router_is_conservative_for_simple_prompts():
    assert not should_plan("Recommend a short fantasy novel.")
    assert should_plan("Find three books, compare them, then recommend the best one.")
