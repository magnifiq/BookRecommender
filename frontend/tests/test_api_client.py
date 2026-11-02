import json

import httpx
import pytest

from frontend.api.client import APIError, BackendAPI


def api_for(handler):
    transport = httpx.MockTransport(handler)
    return BackendAPI("http://backend.test", transport=transport)


@pytest.mark.asyncio
async def test_chat_preserves_conversation_id_in_request_and_response():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"conversation_id": "conv-1", "run_id": "run-1"})

    result = await api_for(handler).chat("Which one is shortest?", "conv-1")
    assert seen == {"path": "/api/chat", "body": {"message": "Which one is shortest?", "conversation_id": "conv-1"}}
    assert result["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_new_chat_omits_missing_conversation_id():
    def handler(request):
        assert json.loads(request.content) == {"message": "Recommend a mystery"}
        return httpx.Response(200, json={"conversation_id": "created"})

    assert (await api_for(handler).chat("Recommend a mystery"))["conversation_id"] == "created"


@pytest.mark.asyncio
async def test_multi_turn_chat_sends_the_same_conversation_id_on_follow_up():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"conversation_id": "conv-2", "run_id": f"run-{len(requests)}"})

    api = api_for(handler)
    first = await api.chat("Recommend science fiction")
    second = await api.chat("Which is shortest?", first["conversation_id"])
    assert requests == [
        {"message": "Recommend science fiction"},
        {"message": "Which is shortest?", "conversation_id": "conv-2"},
    ]
    assert second["conversation_id"] == first["conversation_id"]


@pytest.mark.asyncio
async def test_conversation_history_loading_and_deletion():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.raw_path.decode()))
        if request.method == "GET":
            return httpx.Response(200, json={"conversation": {"conversation_id": "conv 1"}, "messages": [{"role": "user", "content": "Hi"}]})
        return httpx.Response(204)

    api = api_for(handler)
    detail = await api.get_conversation("conv 1")
    await api.delete_conversation("conv 1")
    assert detail["messages"][0]["content"] == "Hi"
    assert calls == [("GET", "/api/conversations/conv%201"), ("DELETE", "/api/conversations/conv%201")]


@pytest.mark.asyncio
async def test_approval_submission_uses_dedicated_endpoint():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"approval": {"status": "approved"}, "run": {"status": "completed"}})

    result = await api_for(handler).resolve_approval("approval-1", "approve")
    assert seen == [("POST", "/api/approvals/approval-1/approve")]
    assert result["approval"]["status"] == "approved"


@pytest.mark.asyncio
async def test_pending_approval_and_saved_preferences_can_be_loaded():
    def handler(request):
        if request.url.path == "/api/approvals":
            return httpx.Response(200, json=[{"approval_id": "approval-1", "tool_name": "demo"}])
        if request.url.path == "/api/approvals/approval-1":
            return httpx.Response(200, json={"approval_id": "approval-1", "arguments": {"text": "example"}})
        return httpx.Response(200, json=[{"preference_id": "pref-1", "category": "favorite_genres", "value": "fantasy"}])

    api = api_for(handler)
    assert (await api.list_approvals())[0]["approval_id"] == "approval-1"
    assert (await api.get_approval("approval-1"))["arguments"]["text"] == "example"
    assert (await api.get_preferences("conv-1"))[0]["value"] == "fantasy"


@pytest.mark.asyncio
async def test_invalid_approval_decision_does_not_call_backend():
    def handler(request):
        pytest.fail("invalid decision reached HTTP client")

    with pytest.raises(ValueError):
        await api_for(handler).resolve_approval("approval-1", "maybe")


@pytest.mark.asyncio
async def test_memory_create_delete_and_clear_routes():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "POST":
            return httpx.Response(201, json={"preference_id": "pref-1"})
        return httpx.Response(204)

    api = api_for(handler)
    await api.create_preference("conv-1", "favorite_genres", "fantasy")
    await api.delete_preference("conv-1", "pref-1")
    await api.clear_preferences("conv-1")
    assert seen == [
        ("POST", "/api/conversations/conv-1/preferences", {"category": "favorite_genres", "value": "fantasy"}),
        ("DELETE", "/api/conversations/conv-1/preferences/pref-1", None),
        ("DELETE", "/api/conversations/conv-1/preferences", None),
    ]


@pytest.mark.asyncio
async def test_trace_and_evaluation_retrieval_and_trigger():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(202, json={"evaluation_id": "eval-1", "status": "queued"})
        return httpx.Response(200, json={"run_id": "run-1", "evaluation_id": "eval-1"})

    api = api_for(handler)
    assert (await api.get_trace("run-1"))["run_id"] == "run-1"
    assert (await api.start_evaluation())["evaluation_id"] == "eval-1"
    assert (await api.get_evaluation("eval-1"))["evaluation_id"] == "eval-1"
    assert seen == [
        ("GET", "/api/traces/run-1"),
        ("POST", "/api/evaluations/run"),
        ("GET", "/api/evaluations/eval-1"),
    ]


@pytest.mark.asyncio
async def test_backend_validation_error_is_presented_without_raw_body():
    def handler(request):
        return httpx.Response(422, json={"detail": [{"loc": ["body"], "msg": "invalid"}]})

    with pytest.raises(APIError) as caught:
        await api_for(handler).chat("hello")
    assert caught.value.status_code == 422
    assert "422" in str(caught.value)
    assert "loc" not in str(caught.value)


@pytest.mark.asyncio
async def test_backend_unavailable_has_friendly_error():
    def handler(request):
        raise httpx.ConnectError("socket details", request=request)

    with pytest.raises(APIError, match="Cannot reach the backend"):
        await api_for(handler).health()


@pytest.mark.asyncio
async def test_backend_server_error_does_not_expose_internal_detail():
    def handler(request):
        return httpx.Response(500, json={"detail": "secret stack trace"})

    with pytest.raises(APIError, match="could not complete") as caught:
        await api_for(handler).health()
    assert "secret" not in str(caught.value)
