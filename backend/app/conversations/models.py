from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: Literal["active", "deleted"] = "active"


class ConversationMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def as_llm_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        return message


class Preference(BaseModel):
    preference_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    category: Literal[
        "favorite_genres", "favorite_authors", "preferred_length",
        "preferred_style", "disliked_genres", "previously_read_books",
    ]
    value: str = Field(min_length=1, max_length=500)
    created_at: datetime = Field(default_factory=utc_now)


class PreferenceCreate(BaseModel):
    category: Preference.model_fields["category"].annotation
    value: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(max_length=100_000)
    conversation_id: str | None = None


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: list[ConversationMessage]


class ChatResponse(BaseModel):
    response: str
    model: str
    run_id: str
    conversation_id: str
    status: str = "completed"
    pending_approval_ids: list[str] = Field(default_factory=list)
