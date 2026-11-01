
from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.book_search import search_books_tool
from app.tools.registry import ToolRegistry
from app.runtime.approvals import ApprovalStore
from app.tools.permission_demos import approval_demo_tool, denied_demo_tool


registry = ToolRegistry()
registry.register(search_books_tool)
registry.register(approval_demo_tool)
registry.register(denied_demo_tool)

trace_collector = TraceCollector()
approval_store = ApprovalStore()

agent = AgentRuntime(
    client=llm_client,
    model=settings.llm_model,
    registry=registry,
    trace_collector=trace_collector,
    tool_execution_timeout_seconds=settings.tool_execution_timeout_seconds,
    approval_store=approval_store,
)
