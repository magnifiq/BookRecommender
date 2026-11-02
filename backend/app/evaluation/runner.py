from collections import Counter, defaultdict
from typing import Awaitable, Callable
from uuid import uuid4

from app.evaluation.models import (
    EvaluationCaseResult,
    EvaluationCategorySummary,
    EvaluationCheck,
    EvaluationMetrics,
    EvaluationRun,
    EvaluationScenario,
)
from app.guardrails import GuardrailBlocked
from app.guardrails.service import Guardrails
from app.runtime.agent import AgentRuntime
from app.runtime.tracing import AgentRun
from app.evaluation.qualitative import QualitativeEvaluator


RuntimeFactory = Callable[[EvaluationScenario], AgentRuntime]
ApprovalResolver = Callable[[EvaluationScenario, AgentRuntime, str], Awaitable[str | None]]


class EvaluationRunner:
    def __init__(self, scenarios: list[EvaluationScenario], runtime_factory: RuntimeFactory,
                 qualitative_evaluator: QualitativeEvaluator | None = None,
                 auto_resolve_approvals: bool = False,
                 approval_resolver: ApprovalResolver | None = None):
        self.scenarios = list(scenarios)
        self.runtime_factory = runtime_factory
        self.qualitative_evaluator = qualitative_evaluator
        self.auto_resolve_approvals = auto_resolve_approvals
        self.approval_resolver = approval_resolver

    async def run(self, evaluation_id: str | None = None, mode: str = "scripted") -> EvaluationRun:
        evaluation = EvaluationRun(evaluation_id=evaluation_id or str(uuid4()), execution_mode=mode)
        observed: list[tuple[EvaluationScenario, EvaluationCaseResult, AgentRun | None]] = []
        for scenario in self.scenarios:
            result, trace = await self._run_case(scenario)
            evaluation.case_results.append(result)
            observed.append((scenario, result, trace))
        evaluation.metrics = self._metrics(observed)
        evaluation.categories = self._categories(evaluation.case_results)
        evaluation.finish()
        return evaluation

    async def _run_case(self, scenario: EvaluationScenario):
        runtime: AgentRuntime | None = None
        trace: AgentRun | None = None
        execution_error: str | None = None
        try:
            runtime = self.runtime_factory(scenario)
            turn_traces = []
            conversation_id = None
            for turn in [scenario.user_input, *scenario.conversation_turns]:
                try:
                    trace = await runtime.run(turn, conversation_id=conversation_id)
                    conversation_id = trace.conversation_id
                except GuardrailBlocked:
                    trace = runtime.trace_collector.get_all_runs()[-1]
                except Exception as error:
                    execution_error = type(error).__name__
                    runs = runtime.trace_collector.get_all_runs()
                    trace = runs[-1] if runs else None
                if trace:
                    turn_traces.append(trace)

                if trace and trace.status == "awaiting_approval":
                    pending = runtime.approval_store.list_pending()
                    if pending:
                        decision = None
                        if self.auto_resolve_approvals:
                            decision = scenario.approval_decision
                        elif self.approval_resolver:
                            decision = await self.approval_resolver(
                                scenario, runtime, pending[0].approval_id
                            )
                        if decision in ("approved", "rejected"):
                            try:
                                trace = await runtime.resolve_approval(pending[0].approval_id, decision)
                                turn_traces[-1] = trace
                            except Exception as error:
                                execution_error = type(error).__name__
                                trace = runtime.trace_collector.get_run(trace.run_id)
                                turn_traces[-1] = trace
            if len(turn_traces) > 1:
                latest = turn_traces[-1]
                trace = latest.model_copy(update={
                    "user_message": scenario.user_input,
                    "events": [event for item in turn_traces for event in item.events],
                    "tool_calls": [tool for item in turn_traces for tool in item.tool_calls],
                    "duration_ms": sum(item.duration_ms or 0 for item in turn_traces),
                })
        except Exception as error:
            execution_error = type(error).__name__
            if runtime:
                runs = runtime.trace_collector.get_all_runs()
                trace = runs[-1] if runs else None

        result = self._compare(scenario, trace, execution_error)
        if self.qualitative_evaluator and trace:
            try:
                result.qualitative_evaluation = await self.qualitative_evaluator.evaluate(
                    scenario, trace.final_response
                )
            except Exception as error:
                result.qualitative_error = type(error).__name__
        return result, trace

    def _compare(self, scenario, trace, execution_error):
        events = trace.events if trace else []
        tools = trace.tool_calls if trace else []
        requested = [tool.tool_name for tool in tools]
        checks: list[EvaluationCheck] = []

        self._add_check(
            checks, "run_status", trace is not None and trace.status == scenario.expected_run_status,
            scenario.expected_run_status, trace.status if trace else "no trace",
        )
        if scenario.expected_tool_calls is not None:
            self._add_check(
                checks, "tool_selection", requested == scenario.expected_tool_calls,
                repr(scenario.expected_tool_calls), repr(requested),
            )
        else:
            self._add_check(
                checks, "no_unexpected_tools", not requested,
                "no tool calls expected", repr(requested),
            )

        if scenario.expected_tool_status is not None:
            actual_status = [tool.status for tool in tools]
            self._add_check(
                checks, "tool_status", actual_status == scenario.expected_tool_status,
                repr(scenario.expected_tool_status), repr(actual_status),
            )
        for index, assertion in enumerate(scenario.tool_result_assertions):
            matching_call = next((call for call in tools if call.tool_name == assertion.tool_name), None)
            actual_value = matching_call.result if matching_call else None
            for component in assertion.result_path.split("."):
                if isinstance(actual_value, dict):
                    actual_value = actual_value.get(component)
                elif isinstance(actual_value, list) and component.isdigit():
                    position = int(component)
                    actual_value = actual_value[position] if position < len(actual_value) else None
                else:
                    actual_value = None
                    break
            self._add_check(
                checks, f"tool_result_{index + 1}", actual_value == assertion.expected_value,
                repr(assertion.expected_value), repr(actual_value),
            )

        retry_count = sum(e.event_type == "llm_retry_scheduled" for e in events)
        if scenario.expected_retry_count is not None:
            self._add_check(
                checks, "retry_count", retry_count == scenario.expected_retry_count,
                scenario.expected_retry_count, retry_count,
            )
        if "retry_success" in scenario.evaluation_criteria:
            self._add_check(
                checks, "retry_success", retry_count > 0 and trace is not None and trace.status == "completed",
                "one or more retries followed by a completed run", f"retries={retry_count}, status={trace.status if trace else 'no trace'}",
            )
        if "retry_limit" in scenario.evaluation_criteria:
            attempts = sum(e.event_type == "llm_request_attempt_started" for e in events)
            expected_attempts = (scenario.expected_retry_count or 0) + 1
            self._add_check(
                checks, "retry_limit", attempts == expected_attempts,
                expected_attempts, attempts,
            )
        if "no_llm_request" in scenario.evaluation_criteria:
            attempts = sum(e.event_type == "llm_request_attempt_started" for e in events)
            self._add_check(checks, "no_llm_request", attempts == 0, 0, attempts)
        if "tool_failure" in scenario.evaluation_criteria:
            failed_tools = sum(tool.status == "failed" for tool in tools)
            self._add_check(checks, "tool_failure", failed_tools > 0 and trace is not None and trace.status == "failed",
                            "failed tool and run", f"failed_tools={failed_tools}, status={trace.status if trace else 'no trace'}")
        if "timeout" in scenario.evaluation_criteria:
            timed_out = any(tool.status == "failed" and tool.error and "timeout" in tool.error.lower() for tool in tools)
            self._add_check(checks, "timeout", timed_out and trace is not None and trace.status == "failed",
                            "timed out tool and failed run", f"timed_out={timed_out}, status={trace.status if trace else 'no trace'}")
        if "denied_not_executed" in scenario.evaluation_criteria:
            denied_ids = {e.details.get("tool_call_id") for e in events
                          if e.event_type == "permission_evaluated" and e.details.get("policy") == "denied"}
            started_ids = {e.details.get("tool_call_id") for e in events if e.event_type == "tool_call_started"}
            self._add_check(checks, "denied_not_executed", not bool(denied_ids & started_ids),
                            "no denied call started", repr(sorted(denied_ids & started_ids)))
        if "approval_before_execution" in scenario.evaluation_criteria:
            call_id = next((tool.tool_call_id for tool in tools if tool.approval_id), None)
            request_index = next((i for i, e in enumerate(events)
                                  if e.event_type == "approval_requested" and e.details.get("tool_call_id") == call_id), None)
            grant_index = next((i for i, e in enumerate(events)
                                if e.event_type == "approval_granted" and e.details.get("tool_call_id") == call_id), None)
            start_index = next((i for i, e in enumerate(events)
                                if e.event_type == "tool_call_started" and e.details.get("tool_call_id") == call_id), None)
            correctly_ordered = request_index is not None and (
                scenario.approval_decision == "rejected" and start_index is None
                or scenario.approval_decision == "approved" and grant_index is not None
                and start_index is not None and request_index < grant_index < start_index
            )
            self._add_check(checks, "approval_before_execution", correctly_ordered,
                            "request < grant < execution (or no execution after rejection)",
                            f"request={request_index}, grant={grant_index}, execution={start_index}")

        if scenario.expected_guardrail and scenario.expected_guardrail != "none":
            mapping = {
                "input_blocked": "input_guardrail_failed",
                "tool_blocked": "tool_guardrail_failed",
                "output_rejected": "output_guardrail_failed",
            }
            expected_event = mapping[scenario.expected_guardrail]
            found = any(e.event_type == expected_event for e in events)
            operation_prevented = True
            if scenario.expected_guardrail == "input_blocked":
                operation_prevented = not any(e.event_type == "llm_request_started" for e in events)
            if scenario.expected_guardrail == "output_rejected":
                operation_prevented = trace is not None and trace.final_response == Guardrails.FALLBACK_RESPONSE
            if scenario.expected_guardrail == "tool_blocked":
                started = {e.details.get("tool_call_id") for e in events if e.event_type == "tool_call_started"}
                failed_calls = {t.tool_call_id for t in tools if t.status == "failed"}
                operation_prevented = not bool(started & failed_calls)
            self._add_check(
                checks, "guardrail_enforcement", found and operation_prevented,
                f"{expected_event} and operation blocked", f"event={found}, blocked={operation_prevented}",
            )

        if scenario.expected_permission:
            policy_events = [e for e in events if e.event_type == "permission_evaluated"]
            policy_match = any(e.details.get("policy") == scenario.expected_permission for e in policy_events)
            execution_started = {e.details.get("tool_call_id") for e in events if e.event_type == "tool_call_started"}
            start_counts = Counter(e.details.get("tool_call_id") for e in events
                                   if e.event_type == "tool_call_started")
            executed_once = all(count == 1 for count in start_counts.values())
            grant_events = [e for e in events if e.event_type == "approval_granted"]
            denial_ids = {e.details.get("tool_call_id") for e in policy_events
                          if e.details.get("policy") == "denied"}
            denied_safe = not bool(denial_ids & execution_started)
            approval_safe = True
            if scenario.expected_permission == "requires_approval":
                approved_ids = {e.details.get("tool_call_id") for e in grant_events}
                approval_safe = execution_started <= approved_ids
                if scenario.approval_decision == "rejected":
                    approval_safe = not bool(execution_started)
                elif scenario.approval_decision == "approved":
                    approval_safe = approval_safe and bool(approved_ids)
            if scenario.expected_permission == "denied":
                approval_safe = approval_safe and not execution_started
            self._add_check(
                checks, "permission_enforcement", policy_match and denied_safe and approval_safe and executed_once,
                scenario.expected_permission, f"policy_match={policy_match}, denied_safe={denied_safe}, approval_safe={approval_safe}",
            )

        if scenario.expected_run_status == "failed" and execution_error:
            # Runtime failure is expected; the public exception text is not retained.
            pass
        passed = all(check.passed for check in checks)
        failure = None if passed else "; ".join(
            f"{check.check_id}: expected {check.expected}, got {check.actual}"
            for check in checks if not check.passed
        )
        actual = (
            f"status={trace.status}; tools={requested}; "
            f"guardrail_failures={sum(e.event_type.endswith('guardrail_failed') for e in events)}"
            if trace else f"no trace; error={execution_error or 'runtime unavailable'}"
        )
        return EvaluationCaseResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            passed=passed,
            expected_behavior=scenario.expected_behavior,
            actual_behavior=actual,
            checks=checks,
            trace_id=trace.run_id if trace else None,
            failure_explanation=failure,
        )

    @staticmethod
    def _add_check(checks, check_id, passed, expected, actual):
        checks.append(EvaluationCheck(
            check_id=check_id,
            passed=bool(passed),
            expected=str(expected),
            actual=str(actual),
            explanation=None if passed else f"{check_id} did not match expectation.",
        ))

    def _metrics(self, observed) -> EvaluationMetrics:
        total = len(observed)
        cases = [item[1] for item in observed]
        traces = [item[2] for item in observed if item[2] is not None]
        requested = [(scenario, trace, [call.tool_name for call in trace.tool_calls] if trace else [])
                     for scenario, _, trace in observed]
        specified = [(scenario, case, trace) for scenario, case, trace in observed
                     if scenario.expected_tool_calls is not None]
        selection_denominator = len(specified)
        selection_numerator = sum(
            ([call.tool_name for call in trace.tool_calls] if trace else []) == scenario.expected_tool_calls
            for scenario, _, trace in specified
        )
        all_actual_tool_names = [name for _, _, names in requested for name in names]
        unexpected = sum(
            sum((Counter(names) - Counter(scenario.expected_tool_calls or [])).values())
            for scenario, _, names in requested
        )
        started_ids = {
            (trace.run_id, e.details.get("tool_call_id"))
            for trace in traces for e in trace.events if e.event_type == "tool_call_started"
        }
        tool_lookup = {(trace.run_id, call.tool_call_id): call for trace in traces for call in trace.tool_calls}
        executed_calls = [tool_lookup[key] for key in started_ids if key in tool_lookup]
        failed_executions = [call for call in executed_calls if call.status == "failed"]
        timeouts = [call for call in failed_executions if call.error and "timeout" in call.error.lower()]
        retried = [trace for trace in traces if any(e.event_type == "llm_retry_scheduled" for e in trace.events)]
        retry_success = sum(trace.status == "completed" for trace in retried)

        rejection_scenarios = [
            (scenario, case) for scenario, case, _ in observed
            if scenario.expected_guardrail in {"input_blocked", "tool_blocked", "output_rejected"}
            or scenario.expected_permission == "denied"
            or (scenario.expected_permission == "requires_approval" and scenario.approval_decision == "rejected")
        ]
        guardrail_cases = [(scenario, case) for scenario, case, _ in observed
                           if scenario.expected_guardrail in {"input_blocked", "tool_blocked", "output_rejected"}]
        permission_cases = [(scenario, case) for scenario, case, _ in observed if scenario.expected_permission]
        denied_calls_by_id = {(trace.run_id, e.details.get("tool_call_id")): (trace, e)
                              for trace in traces for e in trace.events
                              if e.event_type == "permission_evaluated" and e.details.get("policy") == "denied"}
        denied_calls = list(denied_calls_by_id.values())
        denied_executions = sum(
            any(start.event_type == "tool_call_started" and start.details.get("tool_call_id") == event.details.get("tool_call_id")
                for start in trace.events)
            for trace, event in denied_calls
        )
        approval_calls_by_id = {(trace.run_id, event.details.get("tool_call_id")): (trace, event)
                                for trace in traces for event in trace.events
                                if event.event_type == "permission_evaluated" and event.details.get("policy") == "requires_approval"}
        approval_calls = list(approval_calls_by_id.values())
        approval_bypasses = 0
        for trace, event in approval_calls:
            call_id = event.details.get("tool_call_id")
            start_index = next((i for i, e in enumerate(trace.events)
                                if e.event_type == "tool_call_started" and e.details.get("tool_call_id") == call_id), None)
            grant_index = next((i for i, e in enumerate(trace.events)
                                if e.event_type == "approval_granted" and e.details.get("tool_call_id") == call_id), None)
            if start_index is not None and (grant_index is None or grant_index > start_index):
                approval_bypasses += 1

        guardrail_correct = sum(any(c.check_id == "guardrail_enforcement" and c.passed for c in case.checks)
                                for _, case in guardrail_cases)
        permission_correct = sum(any(c.check_id == "permission_enforcement" and c.passed for c in case.checks)
                                for _, case in permission_cases)
        correct_rejections = sum(case.passed for _, case in rejection_scenarios)
        run_durations = [trace.duration_ms for trace in traces if trace.duration_ms is not None]
        tool_durations = [call.duration_ms for call in executed_calls if call.duration_ms is not None]

        rate = lambda numerator, denominator: numerator / denominator if denominator else 0.0
        return EvaluationMetrics(
            scenario_count=total,
            task_completion_rate=rate(sum(case.passed for case in cases), total),
            expected_tool_selection_accuracy=rate(selection_numerator, selection_denominator),
            unexpected_tool_invocation_rate=rate(unexpected, len(all_actual_tool_names)),
            correct_rejection_handling_rate=rate(correct_rejections, len(rejection_scenarios)),
            successful_run_rate=rate(sum(trace.status == "completed" for trace in traces), total),
            tool_execution_failure_rate=rate(len(failed_executions), len(executed_calls)),
            retry_success_rate=rate(retry_success, len(retried)),
            timeout_rate=rate(len(timeouts), len(executed_calls)),
            guardrail_enforcement_rate=rate(guardrail_correct, len(guardrail_cases)),
            permission_enforcement_rate=rate(permission_correct, len(permission_cases)),
            denied_operation_execution_rate=rate(denied_executions, len(denied_calls)),
            approval_bypass_rate=rate(approval_bypasses, len(approval_calls)),
            average_run_duration_ms=sum(run_durations) / len(run_durations) if run_durations else 0.0,
            average_tool_execution_duration_ms=sum(tool_durations) / len(tool_durations) if tool_durations else 0.0,
            average_llm_requests_per_run=sum(sum(e.event_type == "llm_request_attempt_started" for e in t.events) for t in traces) / total if total else 0.0,
            average_tool_calls_per_run=sum(len(names) for _, _, names in requested) / total if total else 0.0,
        )

    @staticmethod
    def _categories(cases: list[EvaluationCaseResult]) -> list[EvaluationCategorySummary]:
        grouped: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
        for case in cases:
            grouped[case.category].append(case)
        return [EvaluationCategorySummary(
            category=category,
            scenario_count=len(items),
            passed_count=sum(case.passed for case in items),
            failed_count=sum(not case.passed for case in items),
            pass_rate=sum(case.passed for case in items) / len(items) if items else 0.0,
        ) for category, items in sorted(grouped.items())]
