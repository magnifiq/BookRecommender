import json
from pathlib import Path

from app.evaluation.models import EvaluationScenario


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "initial.json"


def load_scenarios(path: str | Path = DEFAULT_DATASET) -> list[EvaluationScenario]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [EvaluationScenario.model_validate(item) for item in raw]
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Scenario IDs must be unique")
    return scenarios
