from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.guardrails import GuardrailBlocked, Guardrails
from app.guardrails.models import GuardrailSeverity
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.base import PermissionPolicy, ToolDefinition
from app.tools.registry import ToolRegistry


def response(content=None, tool_calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        model_dump=lambda **_: {"role": "assistant", "content": content},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_input_is_rejected(message):
    results = Guardrails().check_input(message)
    assert any(not result.passed and result.guardrail == "non_empty_input" for result in results)


def test_input_limits_and_repeated_characters_are_configurable():
    checks = Guardrails(max_user_message_length=10, max_repeated_characters=5)
    results = checks.check_input("a" * 11)
    assert any(not r.passed and r.guardrail == "input_length" for r in results)
    assert any(not r.passed and r.guardrail == "repeated_character_run" for r in results)


def test_prompt_injection_wording_is_only_a_risk_signal():
    results = Guardrails().check_input("Explain the phrase: ignore previous instructions")
    signal = next(r for r in results if r.guardrail == "prompt_injection_signal")
    assert signal.passed
    assert signal.severity == GuardrailSeverity.WARNING


class Args(BaseModel):
    query: str


def test_tool_guardrails_reject_malformed_and_unexpected_arguments():
    async def handler(_):
        return None
    definition = ToolDefinition("search", "test", Args, handler)
    checks = Guardrails()
    with pytest.raises(GuardrailBlocked):
        checks.validate_tool_arguments(definition, "not-json")
    with pytest.raises(GuardrailBlocked):
        checks.validate_tool_arguments(definition, '{"query":"x","unexpected":true}')


def test_output_guardrails_return_safe_fallback_for_empty_long_or_internal_output():
    checks = Guardrails(max_output_length=20)
    for value in ("", "x" * 21, "Traceback (most recent call last): secret"):
        safe, results = checks.check_output(value)
        assert safe == checks.FALLBACK_RESPONSE
        assert any(not result.passed for result in results)


@pytest.mark.asyncio
async def test_rejected_input_does_not_make_llm_request():
    async def create(**_):
        pytest.fail("LLM must not be called for rejected input")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = TraceCollector()
    runtime = AgentRuntime(client, "mock", ToolRegistry(), collector)
    with pytest.raises(GuardrailBlocked):
        await runtime.run("  ")
    run = collector.get_all_runs()[0]
    assert run.status == "failed"
    assert "input_guardrail_failed" in [event.event_type for event in run.events]
    assert not any(event.event_type == "llm_request_started" for event in run.events)


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_not_executed_and_are_traced():
    invoked = False
    async def handler(_):
        nonlocal invoked
        invoked = True
    call = SimpleNamespace(id="bad-call", function=SimpleNamespace(name="search", arguments='{"unexpected":true}'))
    tool_message = SimpleNamespace(tool_calls=[call], content=None,
                                   model_dump=lambda **_: {"role": "assistant"})
    client_responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=tool_message, finish_reason="tool_calls")]),
        response("done"),
    ]
    async def create(**_):
        return client_responses.pop(0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    registry = ToolRegistry()
    registry.register(ToolDefinition("search", "test", Args, handler, PermissionPolicy.AUTO))
    collector = TraceCollector()
    run = await AgentRuntime(client, "mock", registry, collector).run("find a book")
    assert not invoked
    assert run.tool_calls[0].status == "failed"
    assert "tool_guardrail_failed" in [event.event_type for event in run.events]


@pytest.mark.asyncio
async def test_invalid_final_output_is_replaced_and_traced():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: _async_response("Traceback (most recent call last): secret")
    )))
    collector = TraceCollector()
    run = await AgentRuntime(client, "mock", ToolRegistry(), collector).run("hello")
    assert run.final_response == Guardrails.FALLBACK_RESPONSE
    assert "output_guardrail_failed" in [event.event_type for event in run.events]


async def _async_response(content):
    return response(content)
