from pydantic import BaseModel, Field, field_validator

from app.tools.base import ToolDefinition


class BookSearchInput(BaseModel):
    query: str = Field(
        description="Book title, author, or topic to search for.",
        min_length=1,
        max_length=500,
    )

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


async def search_books(arguments: BookSearchInput) -> dict:
    # Temporary mock implementation
    return {
        "query": arguments.query,
        "books": [
            {
                "title": "Dune",
                "author": "Frank Herbert",
            },
            {
                "title": "Foundation",
                "author": "Isaac Asimov",
            },
        ],
    }


search_books_tool = ToolDefinition(
    name="search_books",
    description="Search for books by title, author, or topic.",
    args_schema=BookSearchInput,
    handler=search_books,
)
