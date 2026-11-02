import asyncio
import json

import pytest

from app.evaluation.dataset import load_scenarios
from app.evaluation.models import EvaluationScenario
from app.evaluation.models import QualitativeEvaluation
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scripted import make_scripted_runtime
from app.evaluation.service import EvaluationService
from app.runtime.trace_collector import TraceCollector


def test_dataset_loads_approximately_fifty_unique_valid_scenarios():
    scenarios = load_scenarios()
    assert 75 <= len(scenarios) <= 85
    assert len({scenario.scenario_id for scenario in scenarios}) == len(scenarios)
    assert all(isinstance(scenario, EvaluationScenario) for scenario in scenarios)
    assert {"simple_recommendation", "tool_usage", "guardrail_rejection", "approval_required"} <= {
        scenario.category for scenario in scenarios
    }


@pytest.mark.asyncio
async def test_complete_scripted_benchmark_covers_checks_metrics_categories_and_json():
    scenarios = load_scenarios()
    evaluation = await EvaluationRunner(
        scenarios, make_scripted_runtime, auto_resolve_approvals=True
    ).run()
    assert evaluation.status == "completed"
    assert evaluation.metrics.scenario_count == len(scenarios)
    assert evaluation.metrics.task_completion_rate == 1.0
    assert evaluation.metrics.expected_tool_selection_accuracy == 1.0
    assert evaluation.metrics.unexpected_tool_invocation_rate == 0.0
    assert evaluation.metrics.guardrail_enforcement_rate == 1.0
    assert evaluation.metrics.permission_enforcement_rate == 1.0
    assert evaluation.metrics.denied_operation_execution_rate == 0.0
    assert evaluation.metrics.approval_bypass_rate == 0.0
    assert evaluation.metrics.retry_success_rate == pytest.approx(2 / 3)
    assert evaluation.metrics.timeout_rate > 0
    assert evaluation.metrics.average_run_duration_ms > 0
    assert {summary.category for summary in evaluation.categories} >= {
        "multi_step_tool_usage", "llm_retry", "tool_timeout"
    }
    assert all(case.passed for case in evaluation.case_results)
    exported = json.loads(evaluation.to_json())
    assert exported["evaluation_id"] == evaluation.evaluation_id
    assert len(exported["case_results"]) == len(scenarios)


@pytest.mark.asyncio
async def test_empty_dataset_metrics_are_zero_and_report_is_valid():
    evaluation = await EvaluationRunner([], make_scripted_runtime).run()
    assert evaluation.status == "completed"
    assert evaluation.metrics.scenario_count == 0
    assert evaluation.metrics.task_completion_rate == 0
    assert evaluation.metrics.average_run_duration_ms == 0
    assert evaluation.categories == []
    assert json.loads(evaluation.to_json())["case_results"] == []


@pytest.mark.asyncio
async def test_repeated_scripted_subset_has_reproducible_behavior():
    scenarios = load_scenarios()[:6]
    first = await EvaluationRunner(scenarios, make_scripted_runtime).run()
    second = await EvaluationRunner(scenarios, make_scripted_runtime).run()
    assert [(case.scenario_id, case.passed, case.actual_behavior) for case in first.case_results] == [
        (case.scenario_id, case.passed, case.actual_behavior) for case in second.case_results
    ]
    assert first.metrics.task_completion_rate == second.metrics.task_completion_rate
    assert first.categories == second.categories


@pytest.mark.asyncio
async def test_evaluation_runtime_uses_isolated_traces_and_survives_case_failure():
    production_traces = TraceCollector()
    production_run = production_traces.start_run("production")
    scenarios = [next(s for s in load_scenarios() if s.scenario_id == "retry_exhausted_01")]
    eval_collectors = []

    def factory(scenario):
        runtime = make_scripted_runtime(scenario)
        eval_collectors.append(runtime.trace_collector)
        return runtime

    evaluation = await EvaluationRunner(scenarios, factory).run()
    assert evaluation.case_results[0].trace_id is not None
    assert evaluation.case_results[0].passed
    assert production_traces.get_run(production_run.run_id).status == "running"
    assert len(production_traces.get_all_runs()) == 1
    assert eval_collectors[0] is not production_traces


@pytest.mark.asyncio
async def test_background_service_returns_running_handle_and_completes():
    scenario = load_scenarios()[0]
    service = EvaluationService([scenario], make_scripted_runtime)
    pending = service.start()
    assert pending.status == "running"
    for _ in range(100):
        result = service.get(pending.evaluation_id)
        if result.status != "running":
            break
        await asyncio.sleep(0.005)
    assert result.status == "completed"
    assert service.list_recent()[0].evaluation_id == pending.evaluation_id


@pytest.mark.asyncio
async def test_optional_qualitative_scores_do_not_change_deterministic_case_result():
    class MockEvaluator:
        async def evaluate(self, scenario, response):
            return QualitativeEvaluation(
                evaluator_model="test-judge",
                rubric_version="book-quality-v1",
                scores={"relevance": 0.2},
                notes="Subjective example score.",
            )

    evaluation = await EvaluationRunner(
        [load_scenarios()[0]], make_scripted_runtime, MockEvaluator()
    ).run()
    result = evaluation.case_results[0]
    assert result.passed
    assert result.qualitative_evaluation.evaluator_model == "test-judge"
    assert result.qualitative_evaluation.rubric_version == "book-quality-v1"


@pytest.mark.asyncio
async def test_live_style_run_leaves_approval_pending_without_explicit_resolver():
    scenario = next(s for s in load_scenarios() if s.scenario_id == "approval_approved_01")
    result = await EvaluationRunner([scenario], make_scripted_runtime).run(mode="live")
    case = result.case_results[0]
    assert not case.passed
    assert "status=awaiting_approval" in case.actual_behavior


@pytest.mark.asyncio
async def test_explicit_approval_resolver_can_resume_live_style_run():
    scenario = next(s for s in load_scenarios() if s.scenario_id == "approval_approved_01")

    async def resolver(scenario, runtime, approval_id):
        assert runtime.approval_store.get(approval_id).status == "pending"
        return "approved"

    result = await EvaluationRunner(
        [scenario], make_scripted_runtime, approval_resolver=resolver
    ).run(mode="live")
    assert result.case_results[0].passed


def test_duplicate_scenario_ids_are_rejected(tmp_path):
    scenario = load_scenarios()[0].model_dump(mode="json")
    dataset_path = tmp_path / "duplicate.json"
    dataset_path.write_text(json.dumps([scenario, scenario]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(dataset_path)
