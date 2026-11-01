import asyncio

from app.evaluation.models import EvaluationRun, EvaluationScenario
from app.evaluation.runner import EvaluationRunner, RuntimeFactory


class EvaluationService:
    """In-memory background evaluation queue; it never executes caller-supplied code."""

    def __init__(
        self,
        scenarios: list[EvaluationScenario],
        scripted_runtime_factory: RuntimeFactory,
        history_limit: int = 50,
    ):
        self.scenarios = list(scenarios)
        self.runtime_factory = scripted_runtime_factory
        self.history_limit = history_limit
        self._runs: dict[str, EvaluationRun] = {}
        self._tasks: set[asyncio.Task] = set()

    def start(self) -> EvaluationRun:
        run = EvaluationRun(execution_mode="scripted")
        self._runs[run.evaluation_id] = run
        task = asyncio.create_task(self._execute(run.evaluation_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._trim_history()
        return run

    async def _execute(self, evaluation_id: str) -> None:
        try:
            runner = EvaluationRunner(
                self.scenarios,
                self.runtime_factory,
                auto_resolve_approvals=True,
            )
            result = await runner.run(evaluation_id=evaluation_id, mode="scripted")
            self._runs[evaluation_id] = result
        except Exception as error:
            failed = self._runs[evaluation_id]
            failed.status = "failed"
            failed.error = f"Evaluation failed ({type(error).__name__})."
            failed.finish()
            failed.status = "failed"

    def get(self, evaluation_id: str) -> EvaluationRun:
        run = self._runs.get(evaluation_id)
        if run is None:
            raise KeyError(evaluation_id)
        return run

    def list_recent(self, limit: int = 20) -> list[EvaluationRun]:
        return list(reversed(list(self._runs.values())[-limit:]))

    def _trim_history(self) -> None:
        if len(self._runs) <= self.history_limit:
            return
        for evaluation_id, run in list(self._runs.items()):
            if len(self._runs) <= self.history_limit:
                break
            if run.status != "running":
                self._runs.pop(evaluation_id, None)
