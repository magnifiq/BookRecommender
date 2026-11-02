import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.books.openlibrary import BookProviderNotFound, BookProviderUnavailable, OpenLibraryClient
from app.tools.base import ToolDefinition


class SummarizeBookInput(BaseModel):
    identifier: str | None = Field(
        default=None, description="Book title, ISBN, or Open Library work/edition ID."
    )
    supplied_text: str | None = Field(
        default=None, max_length=20000,
        description="Text supplied by the user to summarize instead of fetching a book.",
    )
    length: Literal["short", "standard", "detailed"] = "standard"

    @model_validator(mode="after")
    def require_one_source(self):
        has_identifier = bool(self.identifier and self.identifier.strip())
        has_text = bool(self.supplied_text and self.supplied_text.strip())
        if has_identifier == has_text:
            raise ValueError("provide exactly one of identifier or supplied_text")
        return self


class BookSummary(BaseModel):
    status: Literal["ok", "not_found", "no_description", "unavailable"]
    title: str | None = None
    summary: str | None = None
    basis: Literal["metadata_description", "user_supplied_text"] | None = None
    comprehensive: bool = False
    note: str


def _extractive_summary(text: str, length: str) -> str:
    words_limit = {"short": 35, "standard": 70, "detailed": 110}[length]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        return ""
    stop_words = {"the", "and", "for", "that", "with", "this", "from", "into", "their", "they", "have", "will", "about", "when", "where", "which", "while", "then", "than", "were", "was", "are", "but", "not", "you", "your", "its", "his", "her"}
    tokenized = [[w.casefold() for w in re.findall(r"[A-Za-z0-9']+", sentence)
                  if len(w) > 3 and w.casefold() not in stop_words] for sentence in sentences]
    frequencies = Counter(word for tokens in tokenized for word in tokens)
    scores = [sum(frequencies[word] for word in tokens) / max(len(tokens), 1)
              for tokens in tokenized]
    max_sentences = {"short": 1, "standard": 2, "detailed": 4}[length]
    selected_indexes = sorted(
        sorted(range(len(sentences)), key=lambda index: (-scores[index], index))[:max_sentences]
    )
    selected = []
    used_words = 0
    for index in selected_indexes:
        sentence = sentences[index]
        sentence_words = sentence.split()
        remaining = words_limit - used_words
        if remaining <= 0:
            break
        if len(sentence_words) > remaining:
            sentence = " ".join(sentence_words[:remaining]).rstrip(" ,;:") + "…"
            sentence_words = sentence.split()
        selected.append(sentence)
        used_words += len(sentence_words)
    return " ".join(selected)


def build_summarize_book_tool(provider: OpenLibraryClient) -> ToolDefinition:
    async def summarize_book(arguments: SummarizeBookInput) -> dict:
        if arguments.supplied_text:
            source = arguments.supplied_text
            title = None
            basis = "user_supplied_text"
            note = "Extractive summary of user-supplied text; it does not summarize unprovided book content."
        else:
            try:
                book = await provider.get_details(arguments.identifier or "")
            except BookProviderNotFound:
                return BookSummary(
                    status="not_found", note="No matching catalog record was found."
                ).model_dump(mode="json")
            except BookProviderUnavailable:
                return BookSummary(
                    status="unavailable", note="The catalog is unavailable; no book summary was invented."
                ).model_dump(mode="json")
            if not book:
                return BookSummary(
                    status="not_found", note="No matching catalog record was found."
                ).model_dump(mode="json")
            if not book.description:
                return BookSummary(
                    status="no_description", title=book.title,
                    note="The catalog has no description to summarize. This is not a comprehensive book summary.",
                ).model_dump(mode="json")
            source = book.description
            title = book.title
            basis = "metadata_description"
            note = "Extractive summary based only on the catalog description, not the complete book."

        summary = _extractive_summary(source, arguments.length)
        return BookSummary(
            status="ok" if summary else "no_description",
            title=title,
            summary=summary or None,
            basis=basis if summary else None,
            comprehensive=False,
            note=note,
        ).model_dump(mode="json")

    return ToolDefinition(
        name="summarize_book",
        description=(
            "Create a concise, structured extractive summary from an Open Library "
            "description or user-supplied text. Always state the source basis; "
            "metadata summaries are not comprehensive summaries of a full book."
        ),
        args_schema=SummarizeBookInput,
        handler=summarize_book,
        retry_safe=True,
    )
