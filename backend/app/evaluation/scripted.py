import asyncio
import json
from types import SimpleNamespace

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel

from app.evaluation.models import EvaluationScenario, ScriptedResponse
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.base import PermissionPolicy, ToolDefinition
from app.tools.book_search import BookSearchInput, search_books
from app.tools.registry import ToolRegistry


class DemoText(BaseModel):
    note: str = ""


class DemoValue(BaseModel):
    value: str = ""


class DemoReason(BaseModel):
    reason: str = ""


class ScriptedClient:
    def __init__(self, steps: list[ScriptedResponse]):
        self.steps = list(steps)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **_kwargs):
        if not self.steps:
            raise RuntimeError("Scripted response sequence exhausted")
        step = self.steps.pop(0)
        if step.error:
            request = SimpleNamespace(method="POST", url="https://scripted.invalid")
            if step.error == "connection":
                raise APIConnectionError(request=request)
            if step.error == "timeout":
                raise APITimeoutError(request=request)
            status = {"http_429": 429, "http_500": 500, "http_400": 400}[step.error]
            response = SimpleNamespace(status_code=status, request=request, headers={})
            raise APIStatusError("scripted status", response=response, body=None)

        calls = [SimpleNamespace(
            id=call.tool_call_id,
            type="function",
            function=SimpleNamespace(
                name=call.tool_name,
                arguments=call.raw_arguments or json.dumps(call.arguments),
            ),
        ) for call in step.tool_calls]
        message = SimpleNamespace(
            content=step.content,
            tool_calls=calls,
            model_dump=lambda **_: {
                "role": "assistant",
                "content": step.content,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.function.name, "arguments": call.function.arguments}}
                    for call in calls
                ] or None,
            },
        )
        return SimpleNamespace(choices=[SimpleNamespace(
            message=message, finish_reason="tool_calls" if calls else "stop"
        )])


def make_scripted_runtime(scenario: EvaluationScenario) -> AgentRuntime:
    registry = ToolRegistry()
    registry.register(ToolDefinition("search_books", "Synthetic book search", BookSearchInput, search_books))

    async def approval_handler(args: DemoText):
        return {"accepted_note": args.note}

    async def denied_handler(args: DemoReason):
        return {"unexpected": args.reason}

    async def failure_handler(args: DemoValue):
        outcome = scenario.tool_fixtures.get("failing_demo", "success")
        if outcome == "failure":
            raise RuntimeError("Synthetic tool failure")
        return {"value": args.value}

    async def slow_handler(args: DemoValue):
        outcome = scenario.tool_fixtures.get("slow_demo", "success")
        if outcome == "timeout":
            await asyncio.Event().wait()
        return {"value": args.value}

    registry.register(ToolDefinition(
        "approval_demo", "Harmless approval benchmark tool", DemoText,
        approval_handler, PermissionPolicy.REQUIRES_APPROVAL,
    ))
    registry.register(ToolDefinition(
        "denied_demo", "Harmless denied benchmark tool", DemoReason,
        denied_handler, PermissionPolicy.DENIED,
    ))
    registry.register(ToolDefinition(
        "failing_demo", "Synthetic failure benchmark tool", DemoValue,
        failure_handler,
    ))
    registry.register(ToolDefinition(
        "slow_demo", "Synthetic timeout benchmark tool", DemoValue,
        slow_handler,
    ))

    return AgentRuntime(
        client=ScriptedClient(scenario.scripted_responses),
        model="scripted-evaluation",
        registry=registry,
        trace_collector=TraceCollector(),
        max_llm_retries=2,
        tool_execution_timeout_seconds=0.02,
    )
