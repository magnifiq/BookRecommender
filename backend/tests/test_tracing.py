
from app.runtime.tracing import AgentRun, TraceEvent, ToolCallTrace


def test_agent_run_initialization():
    run = AgentRun(user_message="Find science fiction books")

    assert run.status == "running"
    assert run.run_id
    assert run.events == []
    assert run.tool_calls == []
    assert run.finished_at is None


def test_trace_event_initialization():
    event = TraceEvent(
        event_type="tool_call_started",
        details={"tool_name": "search_books"},
    )

    assert event.event_type == "tool_call_started"
    assert event.details["tool_name"] == "search_books"
    assert event.timestamp is not None


def test_tool_call_trace_initialization():
    trace = ToolCallTrace(
        tool_call_id="call_123",
        tool_name="search_books",
        arguments={"query": "science fiction"},
    )

    assert trace.status == "running"
    assert trace.tool_name == "search_books"
    assert trace.duration_ms is None
