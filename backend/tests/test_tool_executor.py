import pytest
import asyncio
from pydantic import BaseModel

from app.tools.base import ToolDefinition
from app.tools.book_search import search_books_tool
from app.tools.registry import ToolRegistry
from app.tools.executor import ToolExecutor
from app.tools.executor import ToolExecutionTimeout


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


class EmptyArgs(BaseModel):
    pass


class RequiredArgs(BaseModel):
    query: str


def make_executor(handler, timeout=0.05):
    registry = ToolRegistry()
    registry.register(ToolDefinition("synthetic", "test", EmptyArgs, handler))
    return ToolExecutor(registry, timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_tool_completing_before_timeout():
    async def handler(_):
        await asyncio.sleep(0.001)
        return "done"

    assert await make_executor(handler).execute("synthetic", "{}") == "done"


@pytest.mark.asyncio
async def test_timeout_raises_dedicated_error_and_cancels_coroutine():
    cancelled = asyncio.Event()

    async def handler(_):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(ToolExecutionTimeout, match="synthetic"):
        await make_executor(handler, timeout=0.01).execute("synthetic", "{}")
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_ordinary_tool_error_propagates():
    async def handler(_):
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        await make_executor(handler).execute("synthetic", "{}")


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected():
    async def handler(_):
        return None

    registry = ToolRegistry()
    registry.register(ToolDefinition("required", "test", RequiredArgs, handler))
    executor = ToolExecutor(registry)
    with pytest.raises(Exception):
        await executor.execute("required", '{}')
