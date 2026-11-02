import json
from pathlib import Path

from app.evaluation.models import EvaluationScenario


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "initial.json"
ORCHESTRATION_DATASET = Path(__file__).parent / "datasets" / "orchestration.json"


def load_scenarios(path: str | Path | None = None) -> list[EvaluationScenario]:
    paths = [DEFAULT_DATASET, ORCHESTRATION_DATASET] if path is None else [Path(path)]
    raw = [item for dataset_path in paths
           for item in json.loads(Path(dataset_path).read_text(encoding="utf-8"))]
    scenarios = [EvaluationScenario.model_validate(item) for item in raw]
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Scenario IDs must be unique")
    return scenarios
