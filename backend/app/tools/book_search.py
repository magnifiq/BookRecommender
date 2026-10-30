from pydantic import BaseModel, Field

from app.tools.base import ToolDefinition


class BookSearchInput(BaseModel):
    query: str = Field(
        description="Book title, author, or topic to search for."
    )


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