from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.bootstrap import agent, trace_collector
from app.runtime.tracing import AgentRun

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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    response: str
    model: str
    run_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        run = await agent.run(request.message)

        return ChatResponse(
            response=run.final_response or "",
            model=settings.llm_model,
            run_id=run.run_id,
        )

    except Exception:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        )


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
