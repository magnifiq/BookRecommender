from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field

from app.core.config import settings
from app.llm.client import llm_client


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
        completion = await llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are BookRecommender, an AI book assistant. "
                        "You help users discover books, understand "
                        "literature and create original stories."
                    ),
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
        )

        answer = completion.choices[0].message.content

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The model returned an empty response.",
            )

        return ChatResponse(
            response=answer,
            model=settings.llm_model,
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the LLM provider.",
        )

    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="The LLM provider returned an error.",
        )