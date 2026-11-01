import asyncio
from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel

from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.registry import ToolRegistry
from app.tools.base import ToolDefinition
from app.tools.executor import ToolExecutionTimeout


def make_runtime(create, retries=2):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = TraceCollector()
    runtime = AgentRuntime(client, "test-model", ToolRegistry(), collector,
                           max_llm_retries=retries)
    run = collector.start_run("test")
    return runtime, collector, run.run_id


def status_error(status):
    request = SimpleNamespace(method="POST", url="https://example.test")
    response = SimpleNamespace(status_code=status, request=request, headers={})
    return APIStatusError("status failure", response=response, body=None)


def connection_error():
    return APIConnectionError(request=SimpleNamespace(method="POST", url="https://example.test"))


def timeout_error():
    return APITimeoutError(request=SimpleNamespace(method="POST", url="https://example.test"))


@pytest.mark.asyncio
async def test_success_uses_one_attempt(monkeypatch):
    calls = 0
    async def create(**kwargs):
        nonlocal calls
        calls += 1
        return "ok"
    async def no_sleep(_):
        raise AssertionError("sleep should not be called")
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    runtime, _, run_id = make_runtime(create)
    assert await runtime._request_llm([], run_id) == "ok"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [
    connection_error, timeout_error,
    lambda: status_error(429), lambda: status_error(500),
])
async def test_retryable_errors_then_success(monkeypatch, error_factory):
    error = error_factory()
    calls = 0
    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "ok"
    sleeps = []
    async def no_sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    runtime, collector, run_id = make_runtime(create)
    assert await runtime._request_llm([], run_id) == "ok"
    assert calls == 2
    assert sleeps == [1]
    assert [e.event_type for e in collector.get_run(run_id).events].count("llm_retry_scheduled") == 1


@pytest.mark.asyncio
async def test_http_400_is_not_retried(monkeypatch):
    calls = 0
    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise status_error(400)
    async def no_sleep(_):
        raise AssertionError("sleep should not be called")
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    runtime, _, run_id = make_runtime(create)
    with pytest.raises(APIStatusError):
        await runtime._request_llm([], run_id)
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_limit_and_retry_events(monkeypatch):
    calls = 0
    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise connection_error()
    sleeps = []
    async def no_sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    runtime, collector, run_id = make_runtime(create, retries=2)
    with pytest.raises(APIConnectionError):
        await runtime._request_llm([], run_id)
    assert calls == 3
    assert sleeps == [1, 2]
    events = collector.get_run(run_id).events
    assert sum(e.event_type == "llm_retry_scheduled" for e in events) == 2
    assert sum(e.event_type == "llm_request_attempt_started" for e in events) == 3


@pytest.mark.asyncio
async def test_tool_timeout_fails_call_and_run_without_adding_result():
    class Args(BaseModel):
        query: str

    cancelled = asyncio.Event()
    async def handler(_):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
    registry = ToolRegistry()
    registry.register(ToolDefinition("slow", "slow tool", Args, handler))
    tool_call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="slow", arguments='{"query":"x"}')
    )
    assistant = SimpleNamespace(
        tool_calls=[tool_call],
        content=None,
        model_dump=lambda **kwargs: {"role": "assistant"},
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=assistant, finish_reason="tool_calls")])
    seen_messages = []
    async def create(**kwargs):
        seen_messages.append(kwargs["messages"])
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = TraceCollector()
    runtime = AgentRuntime(client, "test", registry, collector,
                           tool_execution_timeout_seconds=0.01)
    with pytest.raises(ToolExecutionTimeout):
        await runtime.run("use slow tool")
    run = collector.get_all_runs()[0]
    assert run.status == "failed"
    assert run.tool_calls[0].status == "failed"
    assert run.tool_calls[0].error.startswith("Tool 'slow'")
    assert run.tool_calls[0].duration_ms is not None
    assert cancelled.is_set()
    assert not any(message.get("role") == "tool" for message in seen_messages[-1])
