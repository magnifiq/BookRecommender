
from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.tools.book_search import search_books_tool
from app.tools.registry import ToolRegistry
from app.runtime.approvals import ApprovalStore
from app.tools.permission_demos import approval_demo_tool, denied_demo_tool
from app.guardrails import Guardrails
from app.evaluation.dataset import load_scenarios
from app.evaluation.scripted import make_scripted_runtime
from app.evaluation.service import EvaluationService


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
    guardrails=Guardrails(
        max_user_message_length=settings.guardrail_max_user_message_length,
        max_output_length=settings.guardrail_max_output_length,
        max_repeated_characters=settings.guardrail_max_repeated_characters,
    ),
)


def make_live_evaluation_runtime(_scenario):
    evaluation_registry = ToolRegistry()
    for tool in registry.get_all():
        evaluation_registry.register(tool)
    return AgentRuntime(
        client=llm_client,
        model=settings.llm_model,
        registry=evaluation_registry,
        trace_collector=TraceCollector(),
        approval_store=ApprovalStore(),
        tool_execution_timeout_seconds=settings.tool_execution_timeout_seconds,
        guardrails=Guardrails(
            max_user_message_length=settings.guardrail_max_user_message_length,
            max_output_length=settings.guardrail_max_output_length,
            max_repeated_characters=settings.guardrail_max_repeated_characters,
        ),
    )


evaluation_service = EvaluationService(
    scenarios=load_scenarios(),
    scripted_runtime_factory=make_scripted_runtime,
)
