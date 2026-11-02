import json
from typing import Any


def history_for_chat(messages: list[dict]) -> list[dict[str, str]]:
    """Show user/assistant text only; tool protocol messages stay in backend history."""
    visible = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            visible.append({"role": role, "content": content})
    return visible


def recent_conversation_choices(conversations: list[dict]) -> list[tuple[str, str]]:
    choices = []
    for item in conversations:
        conversation_id = item.get("conversation_id")
        if not conversation_id:
            continue
        updated = str(item.get("updated_at", ""))[:16].replace("T", " ")
        choices.append((f"{updated or 'Conversation'} · {conversation_id[:8]}", conversation_id))
    return choices


def book_rows_for_trace(trace: dict | None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if not trace:
        return rows
    for tool_call in trace.get("tool_calls") or []:
        result = tool_call.get("result")
        if not isinstance(result, dict):
            continue
        records = result.get("books") or []
        recommendations = result.get("recommendations") or []
        if isinstance(result.get("book"), dict):
            records = [result["book"]]
        if recommendations:
            records = [item.get("book", item) for item in recommendations if isinstance(item, dict)]
        for book in records:
            if not isinstance(book, dict):
                continue
            rows.append([
                book.get("title") or "",
                ", ".join(book.get("authors") or []) if isinstance(book.get("authors"), list) else book.get("authors", ""),
                book.get("publication_year") or "",
                book.get("page_count") or "",
                book.get("description") or "",
                book.get("cover_url") or "",
            ])
    return rows


def tool_result_summary(trace: dict | None) -> str:
    rows = book_rows_for_trace(trace)
    if not rows:
        return "No structured book records were returned for this run."
    return f"Found {len(rows)} structured book record(s). Metadata is shown only when the provider returned it."


def cover_gallery_for_trace(trace: dict | None) -> list[dict[str, str]]:
    """Return provider-supplied cover URLs; never manufacture cover metadata."""
    covers: list[dict[str, str]] = []
    for row in book_rows_for_trace(trace):
        title, _, _, _, _, cover_url = row
        if cover_url:
            covers.append((cover_url, title or "Book cover"))
    return covers


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
