import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.runtime.agent import AgentRuntime
from app.runtime.approvals import ApprovalConflict, ApprovalNotFound, ApprovalStore
from app.runtime.trace_collector import TraceCollector
from app.tools.base import PermissionPolicy, ToolDefinition
from app.tools.registry import ToolRegistry


class Args(BaseModel):
    value: str


def completion(*calls, content=None):
    message = SimpleNamespace(
        tool_calls=list(calls), content=content,
        model_dump=lambda **_: {"role": "assistant", "content": content,
                                "tool_calls": [{"id": c.id, "type": "function",
                                                "function": {"name": c.function.name,
                                                             "arguments": c.function.arguments}}
                                               for c in calls] or None},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")])


def tool_call(call_id, name="sample", value="v"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(
        name=name, arguments=f'{{"value":"{value}"}}'))


def build_runtime(handler, policy, responses, store=None):
    registry = ToolRegistry()
    registry.register(ToolDefinition("sample", "synthetic", Args, handler, policy))
    requests = []
    async def create(**kwargs):
        requests.append(kwargs["messages"])
        return responses.pop(0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    traces = TraceCollector()
    runtime = AgentRuntime(client, "mock", registry, traces, approval_store=store or ApprovalStore())
    return runtime, traces, requests


@pytest.mark.asyncio
async def test_auto_executes_without_approval():
    calls = []
    async def handler(args):
        calls.append(args.value)
        return {"value": args.value}
    runtime, traces, _ = build_runtime(handler, PermissionPolicy.AUTO,
        [completion(tool_call("a")), completion(content="done")])
    run = await runtime.run("go")
    assert calls == ["v"]
    assert run.status == "completed"
    assert not runtime.approval_store.list_pending()
    assert "tool_execution_authorized" in [event.event_type for event in run.events]


@pytest.mark.asyncio
async def test_approval_pauses_then_executes_exactly_once_and_resumes_context():
    executed = []
    async def handler(args):
        executed.append(args.value)
        return {"ok": args.value}
    runtime, traces, requests = build_runtime(handler, PermissionPolicy.REQUIRES_APPROVAL,
        [completion(tool_call("wanted", value="correct")), completion(content="finished")])
    pending = await runtime.run("please")
    approval = runtime.approval_store.list_pending()[0]
    assert pending.status == "awaiting_approval"
    assert executed == []
    assert approval.run_id == pending.run_id
    assert approval.tool_call_id == "wanted"
    assert approval.arguments == {"value": "correct"}
    assert pending.tool_calls[0].status == "awaiting_approval"
    assert "tool_call_started" not in [event.event_type for event in pending.events]

    complete = await runtime.resolve_approval(approval.approval_id, "approved")
    assert complete.status == "completed"
    assert executed == ["correct"]
    resumed = requests[-1]
    assert any(m.get("role") == "assistant" and m.get("tool_calls", [{}])[0].get("id") == "wanted" for m in resumed)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "wanted" for m in resumed)
    assert [event.event_type for event in complete.events].count("approval_granted") == 1
    assert "tool_call_started" in [event.event_type for event in complete.events]
    with pytest.raises(ApprovalConflict):
        await runtime.resolve_approval(approval.approval_id, "approved")
    assert executed == ["correct"]


@pytest.mark.asyncio
async def test_rejection_continues_conversation_without_execution():
    executed = []
    async def handler(args):
        executed.append(args.value)
    runtime, _, requests = build_runtime(handler, PermissionPolicy.REQUIRES_APPROVAL,
        [completion(tool_call("reject-me")), completion(content="continued")])
    pending = await runtime.run("go")
    approval = runtime.approval_store.list_pending()[0]
    run = await runtime.resolve_approval(approval.approval_id, "rejected")
    assert run.status == "completed"
    assert executed == []
    rejected_tool_message = next(m for m in requests[-1] if m.get("role") == "tool")
    assert rejected_tool_message["tool_call_id"] == "reject-me"
    assert "rejected" in rejected_tool_message["content"]


@pytest.mark.asyncio
async def test_denied_policy_never_invokes_handler():
    async def handler(_):
        pytest.fail("denied tool handler must not run")
    runtime, _, _ = build_runtime(handler, PermissionPolicy.DENIED,
        [completion(tool_call("no")), completion(content="denied")])
    run = await runtime.run("try")
    assert run.status == "completed"
    assert run.tool_calls[0].status == "failed"
    assert "tool_execution_denied" in [event.event_type for event in run.events]


@pytest.mark.asyncio
async def test_expired_approval_cannot_execute():
    async def handler(_):
        pytest.fail("expired call must not execute")
    store = ApprovalStore(expiration_seconds=1)
    runtime, _, _ = build_runtime(handler, PermissionPolicy.REQUIRES_APPROVAL,
        [completion(tool_call("expire"))], store)
    await runtime.run("go")
    approval = store.list_pending()[0]
    approval.expires_at = approval.created_at
    with pytest.raises(ApprovalConflict, match="expired"):
        await runtime.resolve_approval(approval.approval_id, "approved")
    assert store.get(approval.approval_id).status == "expired"


@pytest.mark.asyncio
async def test_unknown_approval_id_is_not_found():
    async def handler(_):
        return None
    runtime, _, _ = build_runtime(handler, PermissionPolicy.AUTO, [])
    with pytest.raises(ApprovalNotFound):
        await runtime.resolve_approval("missing", "approved")


@pytest.mark.asyncio
async def test_multiple_runs_keep_separate_pending_tool_contexts():
    seen = []
    async def handler(args):
        seen.append(args.value)
        return args.value
    responses = [completion(tool_call("run-a", value="a")),
                 completion(tool_call("run-b", value="b")),
                 completion(content="a done"), completion(content="b done")]
    registry = ToolRegistry()
    registry.register(ToolDefinition("sample", "synthetic", Args, handler,
                                     PermissionPolicy.REQUIRES_APPROVAL))
    async def create(**kwargs):
        return responses.pop(0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = TraceCollector()
    store = ApprovalStore()
    runtime = AgentRuntime(client, "mock", registry, collector, approval_store=store)
    a, b = await asyncio.gather(runtime.run("a"), runtime.run("b"))
    requests = store.list_pending()
    assert {r.run_id for r in requests} == {a.run_id, b.run_id}
    done_b = await runtime.resolve_approval(next(r.approval_id for r in requests if r.run_id == b.run_id), "approved")
    done_a = await runtime.resolve_approval(next(r.approval_id for r in requests if r.run_id == a.run_id), "approved")
    assert done_a.run_id == a.run_id and done_b.run_id == b.run_id
    assert sorted(seen) == ["a", "b"]


@pytest.mark.asyncio
async def test_tool_failure_after_approval_is_traced_as_failure():
    async def handler(_):
        raise RuntimeError("synthetic failure")
    runtime, _, _ = build_runtime(handler, PermissionPolicy.REQUIRES_APPROVAL,
        [completion(tool_call("fails"))])
    await runtime.run("go")
    approval = runtime.approval_store.list_pending()[0]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        await runtime.resolve_approval(approval.approval_id, "approved")
    run = runtime.trace_collector.get_run(approval.run_id)
    assert run.status == "failed"
    assert run.tool_calls[0].status == "failed"
    assert "approval_granted" in [event.event_type for event in run.events]


@pytest.mark.asyncio
async def test_tool_calls_after_approval_resume_in_original_order():
    order = []
    async def auto_handler(args):
        order.append("auto")
        return "auto result"
    async def approval_handler(args):
        order.append("approved")
        return "approved result"
    registry = ToolRegistry()
    registry.register(ToolDefinition("auto", "automatic", Args, auto_handler, PermissionPolicy.AUTO))
    registry.register(ToolDefinition("needs_ok", "approval", Args, approval_handler,
                                     PermissionPolicy.REQUIRES_APPROVAL))
    calls = [tool_call("auto-id", "auto"), tool_call("approval-id", "needs_ok")]
    responses = [completion(*calls), completion(content="all done")]
    requests = []
    async def create(**kwargs):
        requests.append(kwargs["messages"])
        return responses.pop(0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = TraceCollector()
    store = ApprovalStore()
    runtime = AgentRuntime(client, "mock", registry, collector, approval_store=store)
    waiting = await runtime.run("do both")
    assert waiting.status == "awaiting_approval"
    assert order == ["auto"]
    approval = store.list_pending()[0]
    finished = await runtime.resolve_approval(approval.approval_id, "approved")
    assert finished.status == "completed"
    assert order == ["auto", "approved"]
    resumed_tools = [m["tool_call_id"] for m in requests[-1] if m.get("role") == "tool"]
    assert resumed_tools == ["auto-id", "approval-id"]
