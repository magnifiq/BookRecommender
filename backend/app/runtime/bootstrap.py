
from app.core.config import settings
from app.llm.client import llm_client
from app.runtime.agent import AgentRuntime
from app.runtime.trace_collector import TraceCollector
from app.books.openlibrary import OpenLibraryClient
from app.tools.book_search import build_search_books_tool
from app.tools.book_details import build_book_details_tool
from app.tools.recommend_books import build_recommend_books_tool
from app.tools.summarize_book import build_summarize_book_tool
from app.tools.generate_story import build_generate_story_tool
from app.tools.registry import ToolRegistry
from app.runtime.approvals import ApprovalStore
from app.tools.permission_demos import approval_demo_tool, denied_demo_tool
from app.guardrails import Guardrails
from app.evaluation.dataset import load_scenarios
from app.evaluation.scripted import make_scripted_runtime
from app.evaluation.service import EvaluationService
from app.conversations.repository import SQLiteConversationRepository
from app.conversations.service import ConversationService


registry = ToolRegistry()
book_provider = OpenLibraryClient(
    timeout_seconds=settings.open_library_timeout_seconds,
    min_request_interval_seconds=settings.open_library_min_request_interval_seconds,
    user_agent=settings.open_library_user_agent,
)
registry.register(build_search_books_tool(book_provider))
registry.register(build_book_details_tool(book_provider))
registry.register(build_recommend_books_tool(book_provider))
registry.register(build_summarize_book_tool(book_provider))
registry.register(build_generate_story_tool())
registry.register(approval_demo_tool)
registry.register(denied_demo_tool)

trace_collector = TraceCollector()
approval_store = ApprovalStore()
conversation_repository = SQLiteConversationRepository(settings.conversation_database_path)
conversation_service = ConversationService(
    conversation_repository, max_context_turns=settings.conversation_context_turns
)

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
    conversation_service=conversation_service,
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
