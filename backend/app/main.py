from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.bootstrap import agent



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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer = await agent.run(request.message)

        return ChatResponse(
            response=answer,
            model=settings.llm_model,
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        )