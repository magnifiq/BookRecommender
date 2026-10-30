import pytest

from app.tools.book_search import search_books_tool
from app.tools.registry import ToolRegistry
from app.tools.executor import ToolExecutor


@pytest.fixture
def executor():
    registry = ToolRegistry()
    registry.register(search_books_tool)
    return ToolExecutor(registry)


@pytest.mark.asyncio
async def test_executes_valid_tool(executor):
    result = await executor.execute(
        "search_books",
        '{"query": "science fiction"}',
    )

    assert result["query"] == "science fiction"
    assert len(result["books"]) == 2


@pytest.mark.asyncio
async def test_rejects_invalid_arguments(executor):
    with pytest.raises(Exception):
        await executor.execute(
            "search_books",
            '{"wrong_field": "test"}',
        )


@pytest.mark.asyncio
async def test_rejects_unknown_tool(executor):
    with pytest.raises(KeyError):
        await executor.execute(
            "unknown_tool",
            '{}',
        )