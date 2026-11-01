from typing import Protocol

from app.evaluation.models import EvaluationScenario, QualitativeEvaluation


class QualitativeEvaluator(Protocol):
    """Optional scorer for response quality; its scores never change deterministic pass/fail."""

    async def evaluate(
        self, scenario: EvaluationScenario, response: str | None
    ) -> QualitativeEvaluation:
        """Return subjective scores with evaluator model and rubric provenance."""
        ...
