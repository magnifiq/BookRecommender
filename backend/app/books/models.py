from typing import Literal

from pydantic import BaseModel, Field


class BookRecord(BaseModel):
    provider: str = "Open Library"
    work_id: str | None = None
    edition_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    description: str | None = None
    publication_year: int | None = None
    subjects: list[str] = Field(default_factory=list)
    isbns: list[str] = Field(default_factory=list)
    cover_id: int | None = None
    cover_url: str | None = None
    page_count: int | None = None
    provider_url: str | None = None


class BookSearchResult(BaseModel):
    status: Literal["ok", "unavailable"] = "ok"
    query: str
    books: list[BookRecord] = Field(default_factory=list)
    message: str | None = None


class BookDetailsResult(BaseModel):
    status: Literal["ok", "not_found", "unavailable"]
    book: BookRecord | None = None
    message: str | None = None
