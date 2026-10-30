from app.core.config import settings
from app.llm.client import llm_client
from app.tools.registry import ToolRegistry
from app.tools.book_search import search_books_tool
from app.runtime.agent import AgentRuntime


registry = ToolRegistry()
registry.register(search_books_tool)

agent = AgentRuntime(
    client=llm_client,
    model=settings.llm_model,
    registry=registry,
)