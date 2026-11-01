from enum import StrEnum

from pydantic import BaseModel


class GuardrailStage(StrEnum):
    INPUT = "input"
    TOOL_ARGUMENTS = "tool_arguments"
    OUTPUT = "output"


class GuardrailSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class GuardrailResult(BaseModel):
    passed: bool
    guardrail: str
    severity: GuardrailSeverity
    stage: GuardrailStage
    reason: str | None = None
