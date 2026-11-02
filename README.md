# BookRecommender

This project helps to find you your next favourite book:D

## Book assistant tools

Registered tools are modular builders under `backend/app/tools/`:

- `search_books` finds normalized records by title, author, or subject.
- `get_book_details` retrieves available Work/Edition metadata by title, Open Library ID, or ISBN.
- `recommend_books` searches and ranks real results using transparent genre, author, topic, mood, length, and exclusion rules.
- `summarize_book` makes a short extractive summary of a catalog description or user-supplied text and marks the source. It never claims to summarize the complete book when only a description is available.
- `generate_story` prepares a structured original-writing brief; the agent composes the story in its final LLM response.

The LLM chooses tools and can chain calls across its existing tool rounds. For
example, it can search for a title, inspect the returned work ID, then request
details.

## Guardrails

The runtime checks user messages before making an LLM request, validates tool
arguments against their registered Pydantic schemas before permission checks,
and checks final responses before returning them. Limits can be configured with
`GUARDRAIL_MAX_USER_MESSAGE_LENGTH`, `GUARDRAIL_MAX_OUTPUT_LENGTH`, and
`GUARDRAIL_MAX_REPEATED_CHARACTERS`.

These checks are deterministic safeguards, not a complete prompt-injection or
secret-detection system. Injection-like input is logged as an advisory signal;
it is not proof of malicious intent. Output checks catch selected known patterns
such as configured system instructions, common API-key formats, and Python stack
traces, but cannot reliably identify every disclosure or paraphrase. A failed
output check returns a fixed safe fallback instead of the unchecked response.

## Agent evaluation

The scripted benchmark loads both
`backend/app/evaluation/datasets/initial.json` and
`backend/app/evaluation/datasets/orchestration.json`. Cases have scripted LLM
outputs, static plan proposals, and expected outcomes; the runner uses
synthetic tools and does not contact the LLM provider. Run it with:

```bash
PYTHONPATH=backend python -m app.evaluation.benchmark
```

The API runs the scripted benchmark in the background and stores results in
memory. For a live evaluation, construct `EvaluationRunner` with a runtime
factory that injects the configured client. The runner does not auto-approve
live tool calls; a trusted `approval_resolver` callback is required for a
scenario that requests an approval-gated tool.

## Conversations and preference memory

Conversation history and long-term preferences are stored separately in SQLite.
The default database is `data/conversations.sqlite3`; change it with
`CONVERSATION_DATABASE_PATH`. Conversation IDs are untrusted identifiers, not
authenticated user identities, so preferences are scoped only to that
conversation. Conversation and preference data are deleted together.

## Multi-step orchestration

Complex requests are routed by a conservative deterministic heuristic to an
LLM planner; simple lookups and recommendations stay on the existing runtime
path.

The orchestration benchmark is in
`backend/app/evaluation/datasets/orchestration.json`. Plan metric denominators
are defined as follows:

- Plan validity: valid plans divided by scenarios that attempted planning,
  including deliberately invalid proposals.
- Step completion: completed steps divided by steps in validated plans,
  including blocked and failed steps.
- Dependency enforcement: dependency edges respected (a dependent step starts
  after its source completes, or never starts) divided by all edges in
  validated plans.
- Unnecessary planning: planning attempts in scenarios marked simple divided
  by all such scenarios.
- Duplicate execution: extra tool-start events for the same call divided by
  all tool-start events.
- Successful recovery: expected-recovery scenarios completing after a planned
  tool retry divided by all scenarios marked as expecting recovery.
- Average steps per completed task: completed steps in completed plans divided
  by completed plan tasks.

## Gradio frontend

The frontend is an independent Gradio application in `frontend/`. It talks to
FastAPI only over HTTP; it does not import runtime code or execute tools. Start
the backend and frontend in separate terminals:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

```bash
cd frontend
uv run python app.py
```

Run the
frontend HTTP-client and presentation helper tests from `frontend/` with
`uv run --group dev pytest tests`.
