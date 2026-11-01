from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.books.models import BookSearchResult
from app.books.openlibrary import BookProviderUnavailable, OpenLibraryClient
from app.tools.base import ToolDefinition


class BookSearchInput(BaseModel):
    query: str = Field(
        description="Book title, author name, or subject/genre to search for.",
        min_length=1,
        max_length=200,
    )
    search_by: Literal["title", "author", "subject", "any"] = "any"
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value.strip()


def build_search_books_tool(provider: OpenLibraryClient) -> ToolDefinition:
    async def search_books(arguments: BookSearchInput) -> dict:
        try:
            books = await provider.search(
                query=arguments.query,
                search_by=arguments.search_by,
                limit=arguments.limit,
            )
            result = BookSearchResult(query=arguments.query, books=books)
        except BookProviderUnavailable:
            result = BookSearchResult(
                status="unavailable",
                query=arguments.query,
                message="The book catalog is temporarily unavailable. No results were invented.",
            )
        return result.model_dump(mode="json")

    return ToolDefinition(
        name="search_books",
        description=(
            "Search Open Library by title, author, subject, or general query. "
            "Returns normalized catalog records; an empty list means no matches."
        ),
        args_schema=BookSearchInput,
        handler=search_books,
    )


default_book_provider = OpenLibraryClient()
search_books_tool = build_search_books_tool(default_book_provider)
