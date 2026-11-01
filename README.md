# BookRecommender

This project helps to find you your next favourite book:D

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

The initial benchmark is `backend/app/evaluation/datasets/initial.json`. Each
case has scripted LLM outputs and expected outcomes; the scripted runtime uses
synthetic tools and does not contact the LLM provider. Run it with:

```bash
PYTHONPATH=backend python -m app.evaluation.benchmark
```

The API runs the scripted benchmark in the background and stores results in
memory. For a live evaluation, construct `EvaluationRunner` with a runtime
factory that injects the configured client. The runner does not auto-approve
live tool calls; a trusted `approval_resolver` callback is required for a
scenario that requests an approval-gated tool.

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
