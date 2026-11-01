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

The API also supports background evaluations. Scripted mode is the default;
live mode must be explicitly requested with `{"mode":"live"}` and uses the
configured LLM client. Evaluation runtimes have their own trace collectors and
approval stores. Evaluations are held in memory and disappear on restart.
Only scripted runs apply the dataset's synthetic approval decisions; live runs
leave approval-required calls pending for a real approval action.
