
from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.book_search import search_books_tool
from app.tools.registry import ToolRegistry


registry = ToolRegistry()
registry.register(search_books_tool)

trace_collector = TraceCollector()

agent = AgentRuntime(
    client=llm_client,
    model=settings.llm_model,
    registry=registry,
    trace_collector=trace_collector,
    tool_execution_timeout_seconds=settings.tool_execution_timeout_seconds,
)
