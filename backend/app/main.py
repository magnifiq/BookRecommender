from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.bootstrap import agent, trace_collector, approval_store, evaluation_service, conversation_service
from app.runtime.tracing import AgentRun
from app.runtime.approvals import ApprovalNotFound, ApprovalConflict, ApprovalRequest
from app.guardrails import GuardrailBlocked
from app.evaluation.models import EvaluationRun
from app.conversations.models import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    Preference,
    PreferenceCreate,
)
from app.conversations.repository import ConversationStorageError

import logging

logger = logging.getLogger(__name__)


app = FastAPI(
    title="BookRecommender Agent Platform",
    version="0.1.0",
    description=(
        "An observable AI agent platform for "
        "book discovery and creative writing."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id
        if conversation_id and conversation_service.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        run = await agent.run(request.message, conversation_id=conversation_id)

        return ChatResponse(
            response=run.final_response or "",
            model=settings.llm_model,
            run_id=run.run_id,
            conversation_id=run.conversation_id or "",
            status=run.status,
            pending_approval_ids=[
                call.approval_id for call in run.tool_calls
                if call.status == "awaiting_approval" and call.approval_id
            ],
        )

    except GuardrailBlocked as error:
        raise HTTPException(status_code=400, detail=error.result.reason)
    except HTTPException:
        raise
    except ConversationStorageError:
        logger.exception("Conversation storage failed")
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")
    except Exception:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        )


@app.get("/api/conversations")
async def list_conversations(limit: int = Query(default=50, ge=1, le=100)):
    try:
        return conversation_service.list_conversations(limit)
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    try:
        conversation = conversation_service.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return ConversationDetail(conversation=conversation, messages=conversation_service.history(conversation_id))
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    try:
        if not conversation_service.delete(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return Response(status_code=204)
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


def _require_conversation(conversation_id: str):
    conversation = conversation_service.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.get("/api/conversations/{conversation_id}/preferences", response_model=list[Preference])
async def get_preferences(conversation_id: str):
    try:
        _require_conversation(conversation_id)
        return conversation_service.preferences(conversation_id)
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.post("/api/conversations/{conversation_id}/preferences", response_model=Preference, status_code=201)
async def create_preference(conversation_id: str, request: PreferenceCreate):
    try:
        _require_conversation(conversation_id)
        preference = conversation_service.save_preference(conversation_id, request.category, request.value)
        trace = trace_collector.start_run("Explicit preference update", conversation_id)
        trace_collector.record_event(trace.run_id, "memory_saved", {"conversation_id": conversation_id, "category": request.category})
        trace_collector.complete_run(trace.run_id, "Preference saved.")
        return preference
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.delete("/api/conversations/{conversation_id}/preferences/{preference_id}", status_code=204)
async def delete_preference(conversation_id: str, preference_id: str):
    try:
        _require_conversation(conversation_id)
        if not conversation_service.delete_preference(conversation_id, preference_id):
            raise HTTPException(status_code=404, detail="Preference not found.")
        trace = trace_collector.start_run("Explicit preference deletion", conversation_id)
        trace_collector.record_event(trace.run_id, "memory_deleted", {"conversation_id": conversation_id, "count": 1})
        trace_collector.complete_run(trace.run_id, "Preference deleted.")
        return Response(status_code=204)
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.delete("/api/conversations/{conversation_id}/preferences", status_code=204)
async def clear_preferences(conversation_id: str):
    try:
        _require_conversation(conversation_id)
        deleted = conversation_service.clear_preferences(conversation_id)
        trace = trace_collector.start_run("Explicit preference clear", conversation_id)
        trace_collector.record_event(trace.run_id, "memory_deleted", {"conversation_id": conversation_id, "count": deleted})
        trace_collector.complete_run(trace.run_id, "Preferences cleared.")
        return Response(status_code=204)
    except ConversationStorageError:
        raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable.")


@app.get("/api/traces", response_model=list[AgentRun])
async def list_traces(
    limit: int = Query(default=20, ge=1, le=100),
):
    runs = trace_collector.get_all_runs()

    # Return the most recent runs first.
    return list(reversed(runs[-limit:]))


@app.get("/api/traces/{run_id}", response_model=AgentRun)
async def get_trace(run_id: str):
    try:
        return trace_collector.get_run(run_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Trace '{run_id}' not found.",
        )


@app.get("/api/approvals", response_model=list[ApprovalRequest])
async def list_approvals():
    return approval_store.list_pending()


@app.get("/api/approvals/{approval_id}", response_model=ApprovalRequest)
async def get_approval(approval_id: str):
    try:
        request = approval_store.get(approval_id)
        return request
    except ApprovalNotFound:
        raise HTTPException(status_code=404, detail="Approval request not found.")


@app.post("/api/approvals/{approval_id}/approve")
async def approve_request(approval_id: str):
    return await _resolve_approval(approval_id, "approved")


@app.post("/api/approvals/{approval_id}/reject")
async def reject_request(approval_id: str):
    return await _resolve_approval(approval_id, "rejected")


async def _resolve_approval(approval_id: str, decision: str):
    try:
        run = await agent.resolve_approval(approval_id, decision)
        return {"approval": approval_store.get(approval_id), "run": run}
    except ApprovalNotFound:
        raise HTTPException(status_code=404, detail="Approval request not found.")
    except ApprovalConflict as exc:
        code = 410 if "expired" in str(exc).lower() else 409
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception:
        logger.exception("Approval resolution failed")
        raise HTTPException(status_code=500, detail="Approval resolution failed.")


@app.post("/api/evaluations/run", response_model=EvaluationRun, status_code=202)
async def start_evaluation():
    return evaluation_service.start()


@app.get("/api/evaluations", response_model=list[EvaluationRun])
async def list_evaluations(limit: int = Query(default=20, ge=1, le=100)):
    return evaluation_service.list_recent(limit)


@app.get("/api/evaluations/{evaluation_id}", response_model=EvaluationRun)
async def get_evaluation(evaluation_id: str):
    try:
        return evaluation_service.get(evaluation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Evaluation not found.")


@app.get("/api/evaluations/{evaluation_id}/json")
async def export_evaluation(evaluation_id: str):
    try:
        run = evaluation_service.get(evaluation_id)
        return Response(content=run.to_json(), media_type="application/json")
    except KeyError:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
