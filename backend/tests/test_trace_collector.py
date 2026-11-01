
import pytest

from app.runtime.trace_collector import TraceCollector


def test_start_run():
    collector = TraceCollector()

    run = collector.start_run("Find science fiction books")

    assert run.status == "running"
    assert run.run_id
    assert len(run.events) == 1
    assert run.events[0].event_type == "run_started"


def test_complete_tool_call():
    collector = TraceCollector()
    run = collector.start_run("Find science fiction books")

    tool_call = collector.start_tool_call(
        run_id=run.run_id,
        tool_call_id="call_123",
        tool_name="search_books",
        arguments={"query": "science fiction"},
    )

    result = {
        "books": [
            {"title": "Dune", "author": "Frank Herbert"}
        ]
    }

    completed = collector.complete_tool_call(
        run_id=run.run_id,
        tool_call_id="call_123",
        result=result,
    )

    assert completed.status == "completed"
    assert completed.result == result
    assert completed.finished_at is not None
    assert completed.duration_ms >= 0


def test_fail_tool_call():
    collector = TraceCollector()
    run = collector.start_run("Find books")

    collector.start_tool_call(
        run_id=run.run_id,
        tool_call_id="call_456",
        tool_name="search_books",
        arguments={"query": "fantasy"},
    )

    failed = collector.fail_tool_call(
        run_id=run.run_id,
        tool_call_id="call_456",
        error="Book API unavailable",
    )

    assert failed.status == "failed"
    assert failed.error == "Book API unavailable"
    assert failed.finished_at is not None


def test_complete_run():
    collector = TraceCollector()
    run = collector.start_run("Find books")

    completed = collector.complete_run(
        run_id=run.run_id,
        final_response="Here are some books.",
    )

    assert completed.status == "completed"
    assert completed.final_response == "Here are some books."
    assert completed.finished_at is not None
    assert completed.duration_ms >= 0
    assert completed.events[-1].event_type == "run_completed"


def test_unknown_run_raises_error():
    collector = TraceCollector()

    with pytest.raises(KeyError):
        collector.get_run("unknown-run")


def test_get_all_runs():
    collector = TraceCollector()

    first = collector.start_run("Find fantasy books")
    second = collector.start_run("Find science fiction books")

    runs = collector.get_all_runs()

    assert len(runs) == 2
    assert runs[0].run_id == first.run_id
    assert runs[1].run_id == second.run_id
