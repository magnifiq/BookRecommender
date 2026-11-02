import asyncio
import json
from types import SimpleNamespace

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel

from app.evaluation.models import EvaluationScenario, ScriptedResponse
from app.books.models import BookRecord
from app.books.openlibrary import BookProviderUnavailable
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.base import PermissionPolicy, ToolDefinition
from app.tools.book_search import build_search_books_tool
from app.tools.book_details import build_book_details_tool
from app.tools.recommend_books import build_recommend_books_tool
from app.tools.summarize_book import build_summarize_book_tool
from app.tools.generate_story import build_generate_story_tool
from app.tools.registry import ToolRegistry
from app.conversations.repository import SQLiteConversationRepository
from app.conversations.service import ConversationService


class DemoText(BaseModel):
    note: str = ""


class DemoValue(BaseModel):
    value: str = ""


class DemoReason(BaseModel):
    reason: str = ""


class FixtureBookProvider:
    def __init__(self, scenario: EvaluationScenario):
        self.books = [BookRecord.model_validate(book) for book in scenario.provider_books]
        self.unavailable = scenario.provider_unavailable

    async def search(self, query, search_by="any", limit=5):
        if self.unavailable:
            raise BookProviderUnavailable("Synthetic provider unavailable")
        terms = query.casefold().split()
        matches = []
        for book in self.books:
            if search_by == "title":
                corpus = book.title.casefold()
            elif search_by == "author":
                corpus = " ".join(book.authors).casefold()
            elif search_by == "subject":
                corpus = " ".join(book.subjects).casefold()
            else:
                corpus = " ".join([book.title, *book.authors, *book.subjects, book.description or ""]).casefold()
            if not terms or any(term in corpus for term in terms):
                matches.append(book)
        return matches[:limit]

    async def get_details(self, identifier):
        if self.unavailable:
            raise BookProviderUnavailable("Synthetic provider unavailable")
        for book in self.books:
            if identifier in {book.title, book.work_id, book.edition_id, *book.isbns}:
                return book
        matches = await self.search(identifier, search_by="title", limit=1)
        return matches[0] if matches else None


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
    provider = FixtureBookProvider(scenario)
    registry.register(build_search_books_tool(provider))
    registry.register(build_book_details_tool(provider))
    registry.register(build_recommend_books_tool(provider))
    registry.register(build_summarize_book_tool(provider))
    registry.register(build_generate_story_tool())

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
        conversation_service=ConversationService(SQLiteConversationRepository(":memory:")),
    )
