from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.conversations.repository import ConversationStorageError, SQLiteConversationRepository
from app.conversations.service import ConversationService
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.base import ToolDefinition
from app.tools.registry import ToolRegistry


def response(content=None, tool_calls=()):
    dumped = [{"id": call.id, "type": "function", "function": {
        "name": call.function.name, "arguments": call.function.arguments,
    }} for call in tool_calls]
    message = SimpleNamespace(
        content=content, tool_calls=list(tool_calls),
        model_dump=lambda **_: {"role": "assistant", "content": content,
                                **({"tool_calls": dumped} if dumped else {})},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if dumped else "stop")])


def call(call_id="tool-id"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name="lookup", arguments='{"query":"Dune"}'))


class LookupArgs(BaseModel):
    query: str


@pytest.fixture
def service(tmp_path):
    return ConversationService(SQLiteConversationRepository(tmp_path / "conversations.sqlite3"), max_context_turns=2)


def test_conversations_messages_and_preferences_are_isolated_and_persistent(tmp_path):
    path = tmp_path / "state.sqlite3"
    service = ConversationService(SQLiteConversationRepository(path))
    one, two = service.create(), service.create()
    service.append(one.conversation_id, {"role": "user", "content": "Find Dune"})
    preference = service.save_preference(one.conversation_id, "favorite_genres", "science fiction")

    restarted = ConversationService(SQLiteConversationRepository(path))
    assert restarted.get(one.conversation_id) is not None
    assert [m.content for m in restarted.history(one.conversation_id)] == ["Find Dune"]
    assert restarted.history(two.conversation_id) == []
    assert [p.value for p in restarted.preferences(one.conversation_id)] == ["science fiction"]
    assert restarted.preferences(two.conversation_id) == []
    assert restarted.delete_preference(two.conversation_id, preference.preference_id) is False
    assert restarted.delete(one.conversation_id)
    assert restarted.get(one.conversation_id) is None
    assert restarted.preferences(one.conversation_id) == []


def test_context_truncation_keeps_tool_call_and_result_together(service):
    conversation = service.create()
    cid = conversation.conversation_id
    service.append(cid, {"role": "user", "content": "old"})
    service.append(cid, {"role": "assistant", "content": "old answer"})
    service.append(cid, {"role": "user", "content": "Find a book"})
    assistant_tool = {"id": "wanted", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"Dune"}'}}
    service.append(cid, {"role": "assistant", "content": None, "tool_calls": [assistant_tool]})
    service.append(cid, {"role": "tool", "tool_call_id": "wanted", "content": "Dune result"})
    service.append(cid, {"role": "assistant", "content": "Here it is"})
    service.append(cid, {"role": "user", "content": "Tell me more"})

    messages = service.llm_context(cid)
    assert [m["content"] for m in messages if m["role"] == "user"] == ["Find a book", "Tell me more"]
    call_message = next(m for m in messages if m.get("tool_calls"))
    assert call_message["tool_calls"][0]["id"] == "wanted"
    assert any(m.get("tool_call_id") == "wanted" for m in messages)


@pytest.mark.asyncio
async def test_runtime_resumes_approval_with_persisted_tool_context_and_memory(service):
    conversation = service.create()
    service.save_preference(conversation.conversation_id, "favorite_genres", "science fiction")
    outputs = [response(tool_calls=[call()]), response(content="Dune is a classic."), response(content="It is 412 pages.")]
    sent = []

    async def llm_create(**kwargs):
        sent.append(kwargs["messages"])
        return outputs.pop(0)

    executed = []
    async def handler(args):
        executed.append(args.query)
        return {"title": "Dune"}

    registry = ToolRegistry()
    from app.tools.base import PermissionPolicy
    registry.register(ToolDefinition("lookup", "Lookup", LookupArgs, handler, PermissionPolicy.REQUIRES_APPROVAL))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=llm_create)))
    traces = TraceCollector()
    runtime = AgentRuntime(client, "mock", registry, traces, conversation_service=service)

    pending = await runtime.run("Find a book", conversation.conversation_id)
    assert pending.status == "awaiting_approval"
    assert executed == []
    approval = runtime.approval_store.list_pending()[0]
    done = await runtime.resolve_approval(approval.approval_id, "approved")
    assert done.status == "completed"
    history = service.history(conversation.conversation_id)
    assert next(m for m in history if m.tool_calls).tool_calls[0]["id"] == "tool-id"
    assert next(m for m in history if m.tool_call_id).tool_call_id == "tool-id"
    assert executed == ["Dune"]
    next_run = await runtime.run("How long is it?", conversation.conversation_id)
    assert next_run.final_response == "It is 412 pages."
    prompt = sent[-1]
    assert any(m["role"] == "tool" and m.get("tool_call_id") == "tool-id" for m in prompt)
    assert any(m["role"] == "system" and "science fiction" in (m.get("content") or "") for m in prompt)
    assert next(m for m in history if m.role == "user").content == "Find a book"


def test_explicit_preference_parser_requires_remember_phrase(service):
    assert service.parse_explicit_memory_instruction("I prefer science fiction") is None
    assert service.parse_explicit_memory_instruction("Remember that I prefer science fiction") == [
        ("favorite_genres", "science fiction")
    ]
    assert service.parse_explicit_memory_instruction("Remember that I prefer books under 300 pages") == [
        ("preferred_length", "under 300 pages")
    ]
    assert service.parse_explicit_memory_instruction(
        "Remember that I prefer science fiction and books under 300 pages"
    ) == [("favorite_genres", "science fiction"), ("preferred_length", "under 300 pages")]


@pytest.mark.asyncio
async def test_runtime_saves_only_explicit_memory_and_injects_it_on_later_turn(service):
    prompts = []

    async def llm_create(**kwargs):
        prompts.append(kwargs["messages"])
        return response(content="Here is a science-fiction recommendation.")

    runtime = AgentRuntime(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=llm_create))),
        "mock", ToolRegistry(), TraceCollector(), conversation_service=service,
    )
    saved = await runtime.run("Remember that I prefer science fiction and books under 300 pages.")
    assert saved.status == "completed"
    assert "I’ll remember" in saved.final_response
    assert prompts == []  # Deterministic application handling, no LLM inference or consent.
    assert len(service.preferences(saved.conversation_id)) == 2

    await runtime.run("Recommend something new.", saved.conversation_id)
    assert any(
        message["role"] == "system" and "science fiction" in (message.get("content") or "")
        for message in prompts[0]
    )
    assert any(event.event_type == "memory_retrieved" for event in runtime.trace_collector.get_all_runs()[-1].events)


@pytest.mark.asyncio
async def test_rejected_output_is_not_persisted_in_conversation(service):
    async def llm_create(**_kwargs):
        return response(content="You are BookRecommender, an AI book assistant.")

    runtime = AgentRuntime(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=llm_create))),
        "mock", ToolRegistry(), TraceCollector(), conversation_service=service,
    )
    run = await runtime.run("Hello")
    stored_assistant = [m.content for m in service.history(run.conversation_id) if m.role == "assistant"]
    assert stored_assistant == [runtime.guardrails.FALLBACK_RESPONSE]


@pytest.mark.asyncio
async def test_tool_failure_persists_safe_matching_tool_response(service):
    async def failing_handler(_args):
        raise RuntimeError("private internal exception details")

    registry = ToolRegistry()
    from app.tools.base import PermissionPolicy
    registry.register(ToolDefinition("lookup", "Lookup", LookupArgs, failing_handler, PermissionPolicy.AUTO))
    outputs = [response(tool_calls=[call()])]

    async def llm_create(**_kwargs):
        return outputs.pop(0)

    runtime = AgentRuntime(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=llm_create))),
        "mock", registry, TraceCollector(), conversation_service=service,
    )
    with pytest.raises(RuntimeError, match="private internal"):
        await runtime.run("Find Dune")
    stored = service.history(runtime.trace_collector.get_all_runs()[0].conversation_id)
    assert stored[-1].role == "tool" and stored[-1].tool_call_id == "tool-id"
    assert "private internal" not in stored[-1].content


def test_sqlite_failures_are_wrapped_without_exposing_sqlite_details(tmp_path):
    directory = tmp_path / "not-a-database-file"
    directory.mkdir()
    with pytest.raises(ConversationStorageError, match="Could not initialize"):
        SQLiteConversationRepository(directory)
