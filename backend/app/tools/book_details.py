from pydantic import BaseModel, Field, field_validator

from app.books.models import BookDetailsResult
from app.books.openlibrary import (
    BookProviderNotFound,
    BookProviderUnavailable,
    OpenLibraryClient,
)
from app.tools.base import ToolDefinition


class BookDetailsInput(BaseModel):
    identifier: str = Field(
        description="Open Library work/edition ID, ISBN, or book title.",
        min_length=1,
        max_length=200,
    )

    @field_validator("identifier")
    @classmethod
    def identifier_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value.strip()


def build_book_details_tool(provider: OpenLibraryClient) -> ToolDefinition:
    async def get_book_details(arguments: BookDetailsInput) -> dict:
        try:
            book = await provider.get_details(arguments.identifier)
            result = (
                BookDetailsResult(status="ok", book=book)
                if book else BookDetailsResult(
                    status="not_found", message="No matching book record was found."
                )
            )
        except BookProviderNotFound:
            result = BookDetailsResult(
                status="not_found", message="No matching book record was found."
            )
        except BookProviderUnavailable:
            result = BookDetailsResult(
                status="unavailable", message="The book catalog is temporarily unavailable."
            )
        return result.model_dump(mode="json")

    return ToolDefinition(
        name="get_book_details",
        description="Retrieve available Open Library metadata for a book identifier, ISBN, or title.",
        args_schema=BookDetailsInput,
        handler=get_book_details,
    )
