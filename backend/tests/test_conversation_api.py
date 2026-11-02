import pytest
from fastapi import HTTPException

import app.main as main
from app.conversations.models import PreferenceCreate
from app.conversations.repository import SQLiteConversationRepository
from app.conversations.service import ConversationService
from app.runtime.trace_collector import TraceCollector


@pytest.fixture
def api_state(monkeypatch, tmp_path):
    service = ConversationService(SQLiteConversationRepository(tmp_path / "api.sqlite3"))
    traces = TraceCollector()
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "trace_collector", traces)
    return service, traces


@pytest.mark.asyncio
async def test_preference_endpoints_scope_updates_and_trace_memory_changes(api_state):
    service, traces = api_state
    conversation = service.create()
    preference = await main.create_preference(
        conversation.conversation_id,
        PreferenceCreate(category="favorite_authors", value="Ursula K. Le Guin"),
    )
    assert [item.preference_id for item in await main.get_preferences(conversation.conversation_id)] == [preference.preference_id]
    assert traces.get_all_runs()[-1].events[-2].event_type == "memory_saved"

    response = await main.delete_preference(conversation.conversation_id, preference.preference_id)
    assert response.status_code == 204
    assert await main.get_preferences(conversation.conversation_id) == []
    assert traces.get_all_runs()[-1].events[-2].event_type == "memory_deleted"


@pytest.mark.asyncio
async def test_unknown_conversation_and_preference_return_404(api_state):
    with pytest.raises(HTTPException) as missing:
        await main.get_conversation("unknown")
    assert missing.value.status_code == 404

    service, _ = api_state
    conversation = service.create()
    with pytest.raises(HTTPException) as missing_preference:
        await main.delete_preference(conversation.conversation_id, "unknown")
    assert missing_preference.value.status_code == 404
