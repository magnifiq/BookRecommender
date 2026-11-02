from frontend.components.chat import book_rows_for_trace, cover_gallery_for_trace, history_for_chat, recent_conversation_choices
from frontend.components.evaluations import category_rows, failed_case_rows, metric_rows
from frontend.components.traces import redact, tool_call_rows


def test_conversation_history_keeps_chat_roles_and_hides_protocol_messages():
    messages = [
        {"role": "user", "content": "Find a book"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc-1"}]},
        {"role": "tool", "content": "private protocol result", "tool_call_id": "tc-1"},
        {"role": "assistant", "content": "Here is one."},
    ]
    assert history_for_chat(messages) == [
        {"role": "user", "content": "Find a book"},
        {"role": "assistant", "content": "Here is one."},
    ]


def test_book_results_only_display_available_provider_metadata():
    trace = {"tool_calls": [{"result": {"books": [{"title": "Example", "authors": ["A. Author"], "cover_url": "https://covers.test/1.jpg"}]}}]}
    assert book_rows_for_trace(trace) == [["Example", "A. Author", "", "", "", "https://covers.test/1.jpg"]]
    assert cover_gallery_for_trace(trace) == [("https://covers.test/1.jpg", "Example")]
    assert book_rows_for_trace(None) == []


def test_conversation_choices_require_ids_and_use_updated_time():
    assert recent_conversation_choices([
        {"conversation_id": "abc12345", "updated_at": "2026-10-01T12:30:00Z"},
        {"updated_at": "2026-10-01"},
    ]) == [("2026-10-01 12:30 · abc12345", "abc12345")]


def test_trace_views_redact_secrets_and_preserve_event_details():
    trace = {"tool_calls": [{"tool_name": "search_books", "status": "completed", "arguments": {"api_key": "secret"}, "result": {"title": "Novel"}}]}
    assert redact({"api_key": "secret", "message": "Bearer abc.def.token"}) == {
        "api_key": "[REDACTED]", "message": "Bearer [REDACTED]"
    }
    rows = tool_call_rows(trace)
    assert rows[0][0:2] == ["search_books", "completed"]
    assert "REDACTED" in rows[0][3]


def test_evaluation_tables_cover_metrics_categories_and_failures():
    evaluation = {
        "metrics": {"task_completion_rate": 0.8},
        "categories": [{"category": "search", "scenario_count": 2, "passed_count": 1, "failed_count": 1, "pass_rate": 0.5}],
        "case_results": [{"scenario_id": "case-1", "category": "search", "passed": False, "failure_explanation": "wrong tool"}],
    }
    assert metric_rows(evaluation) == [["Task Completion Rate", 0.8]]
    assert category_rows(evaluation) == [["search", 2, 1, 1, 0.5]]
    assert failed_case_rows(evaluation) == [["case-1", "search", "wrong tool"]]
