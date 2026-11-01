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
