import gradio as gr

from api.client import APIError, BackendAPI
from components.chat import book_rows_for_trace, cover_gallery_for_trace, history_for_chat, recent_conversation_choices
from components.evaluations import category_rows, failed_case_rows, metric_rows
from components.traces import redact, tool_call_rows, trace_event_rows
from config import FrontendConfig


CONFIG = FrontendConfig.from_env()
API = BackendAPI(CONFIG.backend_api_url, CONFIG.request_timeout_seconds)
PREFERENCE_CATEGORIES = [
    "favorite_genres", "favorite_authors", "preferred_length",
    "preferred_style", "disliked_genres", "previously_read_books",
]
CSS = """
.gradio-container {max-width: none !important; width: 100%; min-height: 100vh; box-sizing: border-box;}
.gradio-container .main {width: min(100%, 1440px); margin: 0 auto;}
#app-shell {width: 100%; margin: 0 auto;}
#app-shell .row {align-items: center !important;}
#app-shell .column {justify-content: center;}
#brand {width: 100%; text-align: center; padding: 0.6rem 0 0.2rem;}
#brand h1 {margin-bottom: 0.15rem; letter-spacing: -0.03em;}
#backend-status {text-align: center;}
#app-shell .tab-nav {justify-content: center;}
#app-shell .tabitem {width: 100%;}
.section-note {color: #64748b; font-size: 0.92rem;}
"""


def _error_text(error: Exception) -> str:
    if isinstance(error, APIError):
        return str(error)
    return "The request could not be completed. Please try again."


def _approval_choices(items: list[dict]) -> list[tuple[str, str]]:
    return [(f"{item.get('tool_name', 'Tool')} · {item.get('approval_id', '')[:8]} · expires {str(item.get('expires_at', ''))[:16]}",
             item.get("approval_id")) for item in items if item.get("approval_id")]


def _trace_choices(items: list[dict]) -> list[tuple[str, str]]:
    return [(f"{item.get('status', '')} · {str(item.get('started_at', ''))[:16]} · {item.get('run_id', '')[:8]}",
             item.get("run_id")) for item in items if item.get("run_id")]


def _evaluation_choices(items: list[dict]) -> list[tuple[str, str]]:
    return [(f"{item.get('status', '')} · {str(item.get('started_at', ''))[:16]} · {item.get('evaluation_id', '')[:8]}",
             item.get("evaluation_id")) for item in items if item.get("evaluation_id")]


def _preference_choices(items: list[dict]) -> list[tuple[str, str]]:
    return [(f"{item.get('category', '').replace('_', ' ')}: {item.get('value', '')}", item.get("preference_id"))
            for item in items if item.get("preference_id")]


def create_app(api: BackendAPI = API) -> gr.Blocks:
    async def refresh_backend():
        try:
            await api.health()
            return "🟢 Backend connected"
        except Exception as error:
            return f"🔴 {_error_text(error)}"

    async def refresh_conversations():
        try:
            return gr.update(choices=recent_conversation_choices(await api.list_conversations()), value=None), "Conversations refreshed."
        except Exception as error:
            return gr.update(choices=[], value=None), _error_text(error)

    async def load_conversation(conversation_id):
        if not conversation_id:
            return [], None, "Choose a conversation to load.", [], []
        try:
            result = await api.get_conversation(conversation_id)
            visible = history_for_chat(result.get("messages", []))
            return visible, conversation_id, f"Loaded conversation · {conversation_id[:8]}", [], []
        except Exception as error:
            return [], conversation_id, _error_text(error), [], []

    def new_conversation():
        return [], None, "New conversation. It will be created when you send a message.", [], [], ""

    async def delete_conversation(conversation_id, confirmed, active_conversation_id):
        if not conversation_id:
            return (gr.update(), "Select a conversation first.", gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        if not confirmed:
            return (gr.update(), "Confirm deletion before deleting this conversation.", gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        try:
            await api.delete_conversation(conversation_id)
            choices = recent_conversation_choices(await api.list_conversations())
            clear_active = conversation_id == active_conversation_id
            return (gr.update(choices=choices, value=None), "Conversation and its messages were deleted.",
                    [] if clear_active else gr.update(), None if clear_active else gr.update(),
                    "" if clear_active else gr.update(), [] if clear_active else gr.update(),
                    [] if clear_active else gr.update(), "" if clear_active else gr.update(), False)
        except Exception as error:
            return (gr.update(), _error_text(error), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update())

    async def send_message(message, history, conversation_id):
        current = list(history or [])
        if not isinstance(message, str) or not message.strip():
            yield current, conversation_id, gr.update(interactive=True), "Enter a message first.", [], [], gr.update(interactive=True), gr.update()
            return
        yield current, conversation_id, gr.update(interactive=False), "Contacting the book assistant…", [], [], gr.update(interactive=False), gr.update(interactive=False)
        try:
            result = await api.chat(message.strip(), conversation_id)
            conversation_id = result.get("conversation_id") or conversation_id
            history_note = ""
            try:
                detail = await api.get_conversation(conversation_id) if conversation_id else {"messages": []}
                current = history_for_chat(detail.get("messages", []))
            except Exception:
                current.extend([{"role": "user", "content": message.strip()}])
                if result.get("response"):
                    current.append({"role": "assistant", "content": result["response"]})
                history_note = " Conversation history could not be refreshed; this reply is shown locally."
            run_id = result.get("run_id")
            trace_note = ""
            try:
                trace = await api.get_trace(run_id) if run_id else None
            except Exception:
                trace = None
                trace_note = " Trace details are temporarily unavailable."
            status = result.get("status", "completed")
            if status == "awaiting_approval":
                pending = result.get("pending_approval_ids") or []
                suffix = f" Pending approval: {', '.join(item[:8] for item in pending)}." if pending else " Open the Approvals tab to review the request."
                status_text = "⏸ Waiting for human approval." + suffix
            elif status == "failed":
                status_text = "The run failed. Inspect its trace for step and tool details."
            else:
                status_text = f"Completed · run {str(run_id or '')[:8]}"
            status_text += history_note + trace_note
            books = book_rows_for_trace(trace)
            covers = cover_gallery_for_trace(trace)
            yield current, conversation_id, gr.update(value="", interactive=True), status_text, books, covers, gr.update(interactive=True), gr.update()
        except Exception as error:
            yield current, conversation_id, gr.update(interactive=True), _error_text(error), [], [], gr.update(interactive=True), gr.update()

    async def approval_list():
        try:
            items = await api.list_approvals()
            choices = _approval_choices(items)
            return gr.update(choices=choices, value=choices[0][1] if choices else None), "Pending approvals loaded."
        except Exception as error:
            return gr.update(choices=[], value=None), _error_text(error)

    async def approval_detail(approval_id):
        if not approval_id:
            return {}, "Select a pending approval."
        try:
            return redact(await api.get_approval(approval_id)), "Review the tool and arguments, then choose approve or reject."
        except Exception as error:
            return {}, _error_text(error)

    async def resolve_approval(approval_id, decision, history, active_conversation_id):
        if not approval_id:
            yield {}, "Select a pending approval.", history or [], active_conversation_id, gr.update(), gr.update(interactive=True), gr.update(interactive=True)
            return
        yield {}, "Submitting the decision…", history or [], active_conversation_id, gr.update(), gr.update(interactive=False), gr.update(interactive=False)
        try:
            result = await api.resolve_approval(approval_id, decision)
            run = result.get("run") or {}
            conversation_id = run.get("conversation_id") or active_conversation_id
            refresh_note = ""
            if conversation_id:
                try:
                    detail = await api.get_conversation(conversation_id)
                    history = history_for_chat(detail.get("messages", []))
                except Exception:
                    refresh_note = " Conversation history could not be refreshed."
            try:
                pending = await api.list_approvals()
                choices = _approval_choices(pending)
            except Exception:
                choices = []
                refresh_note += " Pending approvals could not be refreshed."
            final_status = run.get("status", "unknown")
            status_text = f"Decision recorded. Run status: {final_status}.{refresh_note}"
            return_values = (redact(result.get("approval", {})), status_text, history or [],
                             conversation_id, gr.update(choices=choices, value=None),
                             gr.update(interactive=True), gr.update(interactive=True))
            yield return_values
        except Exception as error:
            yield {}, _error_text(error), history or [], active_conversation_id, gr.update(), gr.update(interactive=True), gr.update(interactive=True)

    async def approve_selected(approval_id, history, conversation_id):
        async for update in resolve_approval(approval_id, "approve", history, conversation_id):
            yield update

    async def reject_selected(approval_id, history, conversation_id):
        async for update in resolve_approval(approval_id, "reject", history, conversation_id):
            yield update

    async def load_preferences(conversation_id):
        if not conversation_id:
            return [], gr.update(choices=[], value=None), "Start or load a conversation before managing preferences."
        try:
            items = await api.get_preferences(conversation_id)
            return [[item.get("category"), item.get("value"), item.get("created_at")] for item in items], \
                   gr.update(choices=_preference_choices(items), value=None), f"{len(items)} saved preference(s)."
        except Exception as error:
            return [], gr.update(choices=[], value=None), _error_text(error)

    async def save_preference(conversation_id, category, value):
        if not conversation_id:
            return [], gr.update(choices=[], value=None), "Start or load a conversation first."
        if not value or not value.strip():
            return [], gr.update(choices=[], value=None), "Enter a preference value."
        try:
            await api.create_preference(conversation_id, category, value.strip())
            items = await api.get_preferences(conversation_id)
            return [[item.get("category"), item.get("value"), item.get("created_at")] for item in items], \
                   gr.update(choices=_preference_choices(items), value=None), "Preference saved for this conversation."
        except Exception as error:
            return [], gr.update(choices=[], value=None), _error_text(error)

    async def delete_one_preference(conversation_id, preference_id):
        if not conversation_id or not preference_id:
            return [], gr.update(), "Select a saved preference."
        try:
            await api.delete_preference(conversation_id, preference_id)
            items = await api.get_preferences(conversation_id)
            return [[item.get("category"), item.get("value"), item.get("created_at")] for item in items], \
                   gr.update(choices=_preference_choices(items), value=None), "Preference deleted."
        except Exception as error:
            return [], gr.update(), _error_text(error)

    async def clear_preferences(conversation_id, confirmed):
        if not conversation_id:
            return [], gr.update(), "Start or load a conversation first."
        if not confirmed:
            return [], gr.update(), "Confirm clearing preferences first."
        try:
            await api.clear_preferences(conversation_id)
            return [], gr.update(choices=[], value=None), "Saved preferences cleared."
        except Exception as error:
            return [], gr.update(), _error_text(error)

    async def load_traces():
        try:
            traces = await api.list_traces()
            choices = _trace_choices(traces)
            rows = [[item.get("run_id"), item.get("status"), item.get("duration_ms"),
                     item.get("conversation_id"), item.get("final_response") or item.get("error") or ""]
                    for item in traces]
            return rows, gr.update(choices=choices, value=None), "Recent runs loaded."
        except Exception as error:
            return [], gr.update(choices=[], value=None), _error_text(error)

    async def load_trace(run_id):
        if not run_id:
            return {}, [], [], "Select a run."
        try:
            trace = redact(await api.get_trace(run_id))
            plan = trace.get("execution_plan")
            plan_json = plan if plan else {"message": "No execution plan was created for this run."}
            events = trace_event_rows(trace)
            calls = tool_call_rows(trace)
            return trace, events, calls, plan_json
        except Exception as error:
            return {}, [], [], {"error": _error_text(error)}

    async def load_evaluations():
        try:
            runs = await api.list_evaluations()
            choices = _evaluation_choices(runs)
            return gr.update(choices=choices, value=None), "Evaluation runs loaded."
        except Exception as error:
            return gr.update(choices=[], value=None), _error_text(error)

    async def start_evaluation():
        try:
            run = await api.start_evaluation()
            runs = await api.list_evaluations()
            choices = _evaluation_choices(runs)
            value = run.get("evaluation_id")
            return gr.update(choices=choices, value=value), f"Evaluation queued · {str(value)[:8]}. Refresh to check progress."
        except Exception as error:
            return gr.update(), _error_text(error)

    async def load_evaluation(evaluation_id):
        if not evaluation_id:
            return {}, [], [], [], "Select an evaluation run."
        try:
            run = redact(await api.get_evaluation(evaluation_id))
            metrics = run.get("metrics", {})
            total = metrics.get("scenario_count", 0)
            completed = sum(case.get("passed", False) for case in run.get("case_results", []))
            headline = f"{run.get('status', 'unknown').title()} · {completed}/{total} scenarios passed"
            return run, category_rows(run), metric_rows(run), failed_case_rows(run), headline
        except Exception as error:
            return {}, [], [], [], _error_text(error)

    with gr.Blocks(title="BookRecommender", elem_id="app-shell") as demo:
        gr.Markdown("# 📚 BookRecommender\nA book assistant with transparent runs, approvals, and conversation memory.", elem_id="brand")
        backend_status = gr.Markdown("Checking backend…", elem_id="backend-status")
        with gr.Tabs():
            with gr.Tab("Chat", id="chat"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("### Conversations")
                        conversation_choices = gr.Dropdown(label="Recent conversations", choices=[], value=None)
                        with gr.Row():
                            refresh_conversations_button = gr.Button("Refresh", size="sm")
                            new_conversation_button = gr.Button("New chat", size="sm")
                        conversation_confirmation = gr.Checkbox(label="Confirm deleting this conversation")
                        delete_conversation_button = gr.Button("Delete selected conversation", variant="stop", size="sm")
                        conversation_status = gr.Markdown("Start a new chat or load a recent one.")
                    with gr.Column(scale=3, min_width=480):
                        conversation_state = gr.State(None)
                        chat = gr.Chatbot(label="Book assistant", height=500, layout="bubble")
                        with gr.Row():
                            message = gr.Textbox(label="Message", placeholder="Ask for a recommendation, book details, summary, or original story…", lines=2, scale=8)
                            send_button = gr.Button("Send", variant="primary", scale=1)
                        gr.Examples(
                            examples=[
                                ["Recommend one science-fiction book for someone new to the genre."],
                                ["Find three science-fiction books about artificial intelligence, compare their publication years and descriptions, and recommend the best one for someone new to the genre."],
                            ],
                            inputs=message,
                            label="Try an example prompt",
                            examples_per_page=3,
                        )
                        chat_status = gr.Markdown()
                        gr.Markdown("### Book records returned by tools", elem_classes=["section-note"])
                        book_results = gr.Dataframe(headers=["Title", "Author", "Year", "Pages", "Description", "Cover URL"],
                                                    datatype=["str"] * 6, interactive=False, wrap=True)
                        book_covers = gr.Gallery(label="Available covers", columns=5, height=220, preview=True)

            with gr.Tab("Approvals", id="approvals"):
                gr.Markdown("Review requests generated by registered tools. Approval is submitted only through the backend approval API.")
                with gr.Row():
                    approval_dropdown = gr.Dropdown(label="Pending approval", choices=[], value=None, scale=3)
                    refresh_approvals_button = gr.Button("Refresh approvals")
                approval_status = gr.Markdown()
                approval_json = gr.JSON(label="Tool and validated arguments")
                with gr.Row():
                    approve_button = gr.Button("Approve and resume", variant="primary")
                    reject_button = gr.Button("Reject", variant="stop")

            with gr.Tab("Memory", id="memory"):
                gr.Markdown("Saved reading preferences are separate from chat history and scoped to the active conversation.")
                memory_conversation = gr.Markdown("Active conversation: none")
                with gr.Row():
                    refresh_memory_button = gr.Button("Load saved preferences")
                    preferences_table = gr.Dataframe(headers=["Category", "Value", "Saved at"], datatype=["str", "str", "str"], interactive=False)
                with gr.Row():
                    preference_category = gr.Dropdown(label="Preference type", choices=PREFERENCE_CATEGORIES, value=PREFERENCE_CATEGORIES[0])
                    preference_value = gr.Textbox(label="Preference", placeholder="e.g. science fiction")
                    save_preference_button = gr.Button("Save explicitly", variant="primary")
                with gr.Row():
                    preference_to_delete = gr.Dropdown(label="Preference to delete", choices=[], value=None, scale=3)
                    delete_preference_button = gr.Button("Delete selected")
                    clear_memory_confirmation = gr.Checkbox(label="Confirm clearing all")
                    clear_memory_button = gr.Button("Clear all", variant="stop")
                memory_status = gr.Markdown()

            with gr.Tab("Traces", id="traces"):
                with gr.Row():
                    refresh_traces_button = gr.Button("Load recent runs")
                    trace_status = gr.Markdown()
                trace_runs = gr.Dataframe(headers=["Run ID", "Status", "Duration ms", "Conversation ID", "Response / error"],
                                          datatype=["str", "str", "number", "str", "str"], interactive=False, wrap=True)
                trace_dropdown = gr.Dropdown(label="Inspect run", choices=[], value=None)
                with gr.Row():
                    with gr.Column():
                        trace_events = gr.Dataframe(headers=["Timestamp", "Event", "Details"], datatype=["str", "str", "str"], interactive=False, wrap=True)
                        trace_tools = gr.Dataframe(headers=["Tool", "Status", "Duration ms", "Arguments", "Result", "Error"], datatype=["str"] * 6, interactive=False, wrap=True)
                    with gr.Column():
                        trace_json = gr.JSON(label="Run")
                        plan_json = gr.JSON(label="Execution plan")

            with gr.Tab("Evaluations", id="evaluations"):
                gr.Markdown("Benchmarks run only when requested. Results come from the backend evaluation API.")
                with gr.Row():
                    start_evaluation_button = gr.Button("Run deterministic benchmark", variant="primary")
                    refresh_evaluations_button = gr.Button("Refresh evaluations")
                    evaluation_status = gr.Markdown()
                evaluation_dropdown = gr.Dropdown(label="Evaluation run", choices=[], value=None)
                evaluation_json = gr.JSON(label="Evaluation report")
                evaluation_metrics = gr.Dataframe(headers=["Metric", "Value"], datatype=["str", "str"], interactive=False)
                evaluation_categories = gr.Dataframe(headers=["Category", "Scenarios", "Passed", "Failed", "Pass rate"], datatype=["str", "number", "number", "number", "number"], interactive=False)
                evaluation_failures = gr.Dataframe(headers=["Scenario", "Category", "Failure explanation"], datatype=["str", "str", "str"], interactive=False, wrap=True)

        demo.load(refresh_backend, outputs=backend_status)
        refresh_conversations_button.click(refresh_conversations, outputs=[conversation_choices, conversation_status])
        conversation_choices.change(load_conversation, inputs=conversation_choices,
                                      outputs=[chat, conversation_state, conversation_status, book_results, book_covers])
        new_conversation_button.click(new_conversation,
                                      outputs=[chat, conversation_state, conversation_status, book_results, book_covers, message])
        delete_conversation_button.click(delete_conversation,
                                         inputs=[conversation_choices, conversation_confirmation, conversation_state],
                                         outputs=[conversation_choices, conversation_status, chat, conversation_state,
                                                  chat_status, book_results, book_covers, message,
                                                  conversation_confirmation])
        chat_inputs = [message, chat, conversation_state]
        chat_outputs = [chat, conversation_state, message, chat_status, book_results, book_covers, send_button, new_conversation_button]
        send_button.click(send_message, inputs=chat_inputs, outputs=chat_outputs, concurrency_limit=4)
        message.submit(send_message, inputs=chat_inputs, outputs=chat_outputs, concurrency_limit=4)

        refresh_approvals_button.click(approval_list, outputs=[approval_dropdown, approval_status])
        approval_dropdown.change(approval_detail, inputs=approval_dropdown, outputs=[approval_json, approval_status])
        approve_button.click(approve_selected,
                             inputs=[approval_dropdown, chat, conversation_state],
                             outputs=[approval_json, approval_status, chat, conversation_state,
                                      approval_dropdown, approve_button, reject_button])
        reject_button.click(reject_selected,
                            inputs=[approval_dropdown, chat, conversation_state],
                            outputs=[approval_json, approval_status, chat, conversation_state,
                                     approval_dropdown, approve_button, reject_button])

        memory_conversation_label = lambda value: f"Active conversation: `{value}`" if value else "Active conversation: none"
        conversation_state.change(memory_conversation_label, inputs=conversation_state, outputs=memory_conversation)
        refresh_memory_button.click(load_preferences, inputs=conversation_state,
                                   outputs=[preferences_table, preference_to_delete, memory_status])
        save_preference_button.click(save_preference,
                                    inputs=[conversation_state, preference_category, preference_value],
                                    outputs=[preferences_table, preference_to_delete, memory_status])
        delete_preference_button.click(delete_one_preference,
                                       inputs=[conversation_state, preference_to_delete],
                                       outputs=[preferences_table, preference_to_delete, memory_status])
        clear_memory_button.click(clear_preferences,
                                  inputs=[conversation_state, clear_memory_confirmation],
                                  outputs=[preferences_table, preference_to_delete, memory_status])

        refresh_traces_button.click(load_traces, outputs=[trace_runs, trace_dropdown, trace_status])
        trace_dropdown.change(load_trace, inputs=trace_dropdown,
                              outputs=[trace_json, trace_events, trace_tools, plan_json])
        refresh_evaluations_button.click(load_evaluations, outputs=[evaluation_dropdown, evaluation_status])
        start_evaluation_button.click(start_evaluation, outputs=[evaluation_dropdown, evaluation_status])
        evaluation_dropdown.change(load_evaluation, inputs=evaluation_dropdown,
                                   outputs=[evaluation_json, evaluation_categories, evaluation_metrics,
                                            evaluation_failures, evaluation_status])

    return demo.queue(default_concurrency_limit=8, max_size=64)


def main():
    app = create_app()
    app.launch(server_name=CONFIG.host, server_port=CONFIG.port, share=False, css=CSS, show_error=False)


if __name__ == "__main__":
    main()
