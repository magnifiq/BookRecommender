import re
from collections.abc import Sequence

from app.conversations.models import Conversation, ConversationMessage, Preference
from app.conversations.repository import ConversationRepository


class ConversationService:
    def __init__(self, repository: ConversationRepository, max_context_turns: int = 12):
        if max_context_turns < 1:
            raise ValueError("max_context_turns must be positive")
        self.repository = repository
        self.max_context_turns = max_context_turns

    def create(self) -> Conversation:
        return self.repository.create(Conversation())

    def get(self, conversation_id: str) -> Conversation | None:
        return self.repository.get(conversation_id)

    def list_conversations(self, limit: int = 100):
        return self.repository.list_conversations(limit)

    def delete(self, conversation_id: str) -> bool:
        return self.repository.delete(conversation_id)

    def append(self, conversation_id: str, message: dict) -> ConversationMessage:
        # API SDK message models are converted to plain JSON-compatible values before arrival.
        return self.repository.append(ConversationMessage(
            conversation_id=conversation_id,
            role=message["role"],
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            tool_call_id=message.get("tool_call_id"),
            name=message.get("name"),
        ))

    def history(self, conversation_id: str) -> list[ConversationMessage]:
        return self.repository.messages(conversation_id)

    def llm_context(self, conversation_id: str) -> list[dict]:
        messages = self.history(conversation_id)
        groups: list[list[ConversationMessage]] = []
        for message in messages:
            if message.role == "user" or not groups:
                groups.append([])
            groups[-1].append(message)
        # A user turn includes any assistant tool-call message and every associated tool reply.
        selected = groups[-self.max_context_turns:]
        return [message.as_llm_message() for group in selected for message in group]

    def preferences(self, conversation_id: str) -> list[Preference]:
        return self.repository.preferences(conversation_id)

    def save_preference(self, conversation_id: str, category: str, value: str) -> Preference:
        return self.repository.save_preference(Preference(
            conversation_id=conversation_id, category=category, value=value.strip()
        ))

    def delete_preference(self, conversation_id: str, preference_id: str) -> bool:
        return self.repository.delete_preference(conversation_id, preference_id)

    def clear_preferences(self, conversation_id: str) -> int:
        return self.repository.clear_preferences(conversation_id)

    def preference_context(self, conversation_id: str) -> str | None:
        preferences = self.preferences(conversation_id)
        if not preferences:
            return None
        grouped: dict[str, list[str]] = {}
        for preference in preferences:
            grouped.setdefault(preference.category, []).append(preference.value)
        lines = [f"{key}: {', '.join(values)}" for key, values in grouped.items()]
        return ("Saved preferences explicitly provided by the user for this conversation: "
                + "; ".join(lines)
                + ". Use them when relevant; the current request overrides any saved preference.")

    def parse_explicit_memory_instruction(self, message: str) -> list[tuple[str, str]] | None:
        """Parse a narrow allowlist of explicit remember commands; never infer preferences."""
        match = re.match(r"^\s*remember\s+(?:that\s+)?i\s+(.+?)\s*[.!]?\s*$", message, re.I)
        if not match:
            return None
        statement = match.group(1).strip()
        combined = re.fullmatch(
            r"(?:prefer|like)\s+(.+?)\s+and\s+books?\s+under\s+(\d+)\s+pages",
            statement, re.I,
        )
        if combined:
            return [("favorite_genres", combined.group(1).strip()),
                    ("preferred_length", f"under {combined.group(2)} pages")]
        length = re.fullmatch(r"(?:prefer|like)\s+(?:reading\s+)?books?\s+under\s+(\d+)\s+pages", statement, re.I)
        if length:
            return [("preferred_length", f"under {length.group(1)} pages")]
        patterns: Sequence[tuple[str, str]] = (
            (r"(?:prefer|like)\s+(?:writing\s+)?style\s+(.+)", "preferred_style"),
            (r"(?:prefer|like)\s+(?:reading\s+)?(.+?)\s+books?\s+under\s+(\d+)\s+pages", "preferred_length"),
            (r"(?:prefer|like)\s+(?:reading\s+)?(.+?)\s+books?", "preferred_length"),
            (r"(?:prefer|like)\s+(?:reading\s+)?(?:the\s+)?(?:genre\s+)?(.+?)\s*(?:books?)?", "favorite_genres"),
            (r"(?:favorite|favourite)\s+author\s+(?:is\s+)?(.+)", "favorite_authors"),
            (r"(?:dislike|hate)\s+(.+?)\s*(?:books?)?", "disliked_genres"),
            (r"(?:have\s+read|already\s+read)\s+(.+)", "previously_read_books"),
        )
        for pattern, category in patterns:
            found = re.fullmatch(pattern, statement, re.I)
            if found:
                value = " ".join(part for part in found.groups() if part).strip(" .,!?")
                if value:
                    return [(category, value)]
        return None
