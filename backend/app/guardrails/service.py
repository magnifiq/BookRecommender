import json
import re
from pydantic import BaseModel

from app.guardrails.models import (
    GuardrailResult,
    GuardrailSeverity,
    GuardrailStage,
)


class GuardrailBlocked(ValueError):
    def __init__(self, result: GuardrailResult):
        self.result = result
        super().__init__(result.reason or "Guardrail check failed")


class Guardrails:
    """Deterministic lifecycle checks; these checks are signals, not a security proof."""

    FALLBACK_RESPONSE = "I can’t provide that response. Please try rephrasing your request."

    def __init__(
        self,
        max_user_message_length: int = 4000,
        max_output_length: int = 12000,
        max_repeated_characters: int = 100,
        max_tool_argument_bytes: int = 16000,
    ):
        self.max_user_message_length = max_user_message_length
        self.max_output_length = max_output_length
        self.max_repeated_characters = max_repeated_characters
        self.max_tool_argument_bytes = max_tool_argument_bytes

    def check_input(self, message: str) -> list[GuardrailResult]:
        valid_text = message if isinstance(message, str) else ""
        results = [self._result(
            passed=bool(valid_text and valid_text.strip()),
            guardrail="non_empty_input",
            stage=GuardrailStage.INPUT,
            reason="Message must not be empty or whitespace-only.",
        )]
        results.append(self._result(
            passed=len(valid_text) <= self.max_user_message_length,
            guardrail="input_length",
            stage=GuardrailStage.INPUT,
            reason="Message exceeds the configured length limit.",
        ))
        repeated = self.max_repeated_characters > 0 and re.search(
            rf"(.)\1{{{self.max_repeated_characters - 1},}}", valid_text
        )
        results.append(self._result(
            passed=not bool(repeated),
            guardrail="repeated_character_run",
            stage=GuardrailStage.INPUT,
            reason="Message contains an unusually long repeated-character run.",
        ))

        injection_pattern = re.compile(
            r"\b(ignore|disregard|override)\b.{0,80}\b(previous|prior|above|system)\b.{0,40}\b(instructions?|prompt)\b"
            r"|\b(reveal|show|print|repeat|disclose)\b.{0,60}\b(system prompt|hidden instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        )
        risk = bool(injection_pattern.search(valid_text))
        # This is deliberately advisory: quoted, educational, or benign mentions
        # can match simple patterns and are not sufficient grounds for rejection.
        results.append(GuardrailResult(
            passed=True,
            guardrail="prompt_injection_signal",
            severity=GuardrailSeverity.WARNING if risk else GuardrailSeverity.INFO,
            stage=GuardrailStage.INPUT,
            reason="Instruction-override wording detected; treated as a risk signal only." if risk else None,
        ))
        return results

    def validate_tool_arguments(self, tool, raw_arguments: str) -> tuple[BaseModel, list[GuardrailResult]]:
        if not isinstance(raw_arguments, str):
            result = self._result(False, "tool_argument_json", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments must be valid JSON text.")
            raise GuardrailBlocked(result)
        try:
            argument_bytes = raw_arguments.encode("utf-8")
        except UnicodeEncodeError:
            argument_bytes = b""
        if len(argument_bytes) > self.max_tool_argument_bytes:
            result = self._result(False, "tool_argument_size", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments exceed the configured size limit.")
            raise GuardrailBlocked(result)
        try:
            data = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            result = self._result(False, "tool_argument_json", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments must be valid JSON.")
            raise GuardrailBlocked(result)
        if not isinstance(data, dict):
            result = self._result(False, "tool_argument_shape", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments must be a JSON object.")
            raise GuardrailBlocked(result)

        fields = tool.args_schema.model_fields
        allowed = set(fields)
        for name, field in fields.items():
            if isinstance(field.alias, str):
                allowed.add(field.alias)
            if isinstance(field.validation_alias, str):
                allowed.add(field.validation_alias)
        if set(data) - allowed:
            result = self._result(False, "unexpected_tool_arguments", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments contain unexpected fields.")
            raise GuardrailBlocked(result)
        try:
            model = tool.args_schema.model_validate(data, extra="forbid", strict=True)
        except Exception:
            result = self._result(False, "tool_argument_schema", GuardrailStage.TOOL_ARGUMENTS,
                                  "Tool arguments do not satisfy the registered schema.")
            raise GuardrailBlocked(result)
        return model, [self._result(True, "tool_argument_schema", GuardrailStage.TOOL_ARGUMENTS)]

    def check_output(self, response: str | None) -> tuple[str, list[GuardrailResult]]:
        value = response if isinstance(response, str) else ""
        results = [self._result(
            passed=bool(value.strip()),
            guardrail="non_empty_output",
            stage=GuardrailStage.OUTPUT,
            reason="Final response must not be empty.",
        )]
        results.append(self._result(
            passed=len(value) <= self.max_output_length,
            guardrail="output_length",
            stage=GuardrailStage.OUTPUT,
            reason="Final response exceeds the configured length limit.",
        ))
        internal_patterns = (
            r"You are BookRecommender, an AI book assistant",
            r"Use available tools when they help answer the user's request",
            r"\b(?:sk|rk)-[A-Za-z0-9_-]{12,}\b",
            r"\b(?:OPENAI_API_KEY|LLM_API_KEY|API_KEY)\s*[:=]\s*\S+",
            r"\b(?:api[_ -]?key|access token)\s*[:=]\s*\S{8,}",
            r"\b(?:system prompt|hidden instructions?)\s*[:=]\s*.{20,}",
            r"\b(?:APIConnectionError|APITimeoutError|APIStatusError|ToolExecutionTimeout|ValidationError)\b",
            r"Traceback \(most recent call last\):",
            r"File \"[^\"]+\.py\", line \d+",
        )
        exposes_internal = any(re.search(p, value, re.IGNORECASE) for p in internal_patterns)
        results.append(self._result(
            passed=not exposes_internal,
            guardrail="internal_information_disclosure",
            stage=GuardrailStage.OUTPUT,
            reason="Response appears to contain internal instructions, credentials, or a stack trace." if exposes_internal else None,
        ))
        if any(not result.passed for result in results):
            return self.FALLBACK_RESPONSE, results
        return value, results

    @staticmethod
    def _result(passed: bool, guardrail: str, stage: GuardrailStage,
                reason: str | None = None) -> GuardrailResult:
        return GuardrailResult(
            passed=passed,
            guardrail=guardrail,
            severity=GuardrailSeverity.INFO if passed else GuardrailSeverity.ERROR,
            stage=stage,
            reason=None if passed else reason,
        )
