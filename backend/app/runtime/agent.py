
import json, asyncio, time
from dataclasses import dataclass
from datetime import datetime, timezone

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from app.runtime.trace_collector import TraceCollector
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.base import PermissionPolicy
from app.runtime.approvals import ApprovalStore, ApprovalConflict
from app.guardrails import GuardrailBlocked, Guardrails
from app.guardrails.models import GuardrailResult, GuardrailStage
from app.orchestration.engine import PlanExecutionEngine, StepExecutionResult
from app.orchestration.models import (
    ExecutionPlan, PlanLimits, PlanStatus, PlanStep, StepStatus, StepType,
)
from app.orchestration.planner import LLMPlanPlanner, should_plan
from app.orchestration.validator import PlanValidator
from app.tools.executor import ToolExecutionTimeout


@dataclass
class _RunState:
    messages: list
    conversation_id: str | None = None
    tool_rounds: int = 0
    tool_calls: list | None = None
    next_tool_index: int = 0
    approval_id: str | None = None
    decision: str | None = None


@dataclass
class _PlanRunState:
    plan: ExecutionPlan
    outputs: dict
    preference_context: str | None = None
    active_elapsed_seconds: float = 0.0
    tool_calls_used: int = 0
    pending_approval_id: str | None = None
    approval_decision: str | None = None


class AgentRuntime:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        registry: ToolRegistry,
        trace_collector: TraceCollector,
        max_tool_rounds: int = 5,
        max_llm_retries: int = 2,
        tool_execution_timeout_seconds: float = 10.0,
        approval_store: ApprovalStore | None = None,
        guardrails: Guardrails | None = None,
        conversation_service=None,
        plan_limits: PlanLimits | None = None,
        plan_planner=None,
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry, tool_execution_timeout_seconds)
        self.trace_collector = trace_collector
        self.max_tool_rounds = max_tool_rounds
        self.max_llm_retries = max_llm_retries
        self.approval_store = approval_store or ApprovalStore()
        self.guardrails = guardrails or Guardrails()
        self.conversation_service = conversation_service
        self.plan_limits = plan_limits or PlanLimits()
        self.plan_planner = plan_planner or LLMPlanPlanner()
        self.plan_validator = PlanValidator(registry, max_steps=self.plan_limits.max_steps)
        self.plan_engine = PlanExecutionEngine(self.plan_limits)
        self._run_states: dict[str, _RunState] = {}
        self._plan_states: dict[str, _PlanRunState] = {}
        self.approval_store.on_expire = self._handle_approval_expiry

    def _handle_approval_expiry(self, request):
        self.trace_collector.record_event(request.run_id, "approval_expired", {
            "approval_id": request.approval_id,
            "tool_call_id": request.tool_call_id,
        })
        state = self._run_states.pop(request.run_id, None)
        if state and state.approval_id == request.approval_id:
            self.trace_collector.fail_tool_call(
                request.run_id, request.tool_call_id, "Approval request expired"
            )
            self.trace_collector.fail_run(request.run_id, "Approval request expired")
        plan_state = self._plan_states.pop(request.run_id, None)
        if plan_state and plan_state.pending_approval_id == request.approval_id:
            step = next((item for item in plan_state.plan.steps
                         if item.approval_id == request.approval_id), None)
            if step:
                step.status = StepStatus.FAILED
                step.error = "Approval request expired."
                self.trace_collector.fail_tool_call(
                    request.run_id, request.tool_call_id, "Approval request expired"
                )
                self.trace_collector.record_event(request.run_id, "step_failed", {
                    "plan_id": plan_state.plan.plan_id, "step_id": step.step_id,
                    "error_type": "ApprovalExpired",
                })
                if self.conversation_service is not None:
                    conversation_id = self.trace_collector.get_run(request.run_id).conversation_id
                    if conversation_id:
                        self.conversation_service.append(conversation_id, {
                            "role": "tool", "tool_call_id": step.tool_call_id,
                            "content": json.dumps({"error": "Approval request expired."}),
                        })
            plan_state.plan.status = PlanStatus.FAILED
            plan_state.plan.error = "Approval request expired."
            self.trace_collector.record_event(request.run_id, "plan_failed", {
                "plan_id": plan_state.plan.plan_id, "reason": "approval_expired",
            })
            self.trace_collector.fail_run(request.run_id, "Approval request expired")
            plan_state.plan.completed_at = self.trace_collector.get_run(request.run_id).finished_at
            self._run_states.pop(request.run_id, None)

    async def run(self, user_message: str, conversation_id: str | None = None):
        memory_context = None
        if self.conversation_service is not None:
            if conversation_id is None:
                conversation = self.conversation_service.create()
                conversation_id = conversation.conversation_id
                # The run does not exist yet, so emit this just after it is created.
                created = True
            else:
                conversation = self.conversation_service.get(conversation_id)
                if conversation is None:
                    raise KeyError("Unknown conversation")
                created = False
        else:
            created = False
        run = self.trace_collector.start_run(user_message, conversation_id)
        run_id = run.run_id
        if conversation_id:
            self.trace_collector.record_event(run_id, "conversation_created" if created else "conversation_resumed", {
                "conversation_id": conversation_id,
            })
        results = self.guardrails.check_input(user_message)
        self._record_guardrail_results(run_id, results)
        failed = next((result for result in results if not result.passed), None)
        if failed:
            self.trace_collector.fail_run(run_id, failed.reason or "Input rejected by guardrails")
            raise GuardrailBlocked(failed)

        if self.conversation_service is not None:
            explicit_preferences = self.conversation_service.parse_explicit_memory_instruction(user_message)
            if explicit_preferences:
                saved = [self.conversation_service.save_preference(conversation_id, category, value)
                         for category, value in explicit_preferences]
                response_text = "I’ll remember these preferences for this conversation: " + "; ".join(
                    f"{preference.category.replace('_', ' ')}: {preference.value}" for preference in saved
                ) + "."
                self.conversation_service.append(conversation_id, {"role": "user", "content": user_message})
                self.conversation_service.append(conversation_id, {"role": "assistant", "content": response_text})
                self.trace_collector.record_event(run_id, "memory_saved", {
                    "conversation_id": conversation_id,
                    "categories": [category for category, _ in explicit_preferences],
                    "count": len(saved),
                })
                return self.trace_collector.complete_run(run_id, response_text)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are BookRecommender, an AI book assistant. "
                    "Use available tools when they help answer "
                    "the user's request."
                ),
            },
        ]
        if self.conversation_service is not None:
            history = self.conversation_service.llm_context(conversation_id)
            self.trace_collector.record_event(run_id, "conversation_context_loaded", {
                "conversation_id": conversation_id, "message_count": len(history),
            })
            messages.extend(history)
            memory_context = self.conversation_service.preference_context(conversation_id)
            self.trace_collector.record_event(run_id, "memory_retrieved", {
                "conversation_id": conversation_id,
                "preference_count": len(self.conversation_service.preferences(conversation_id)),
            })
            if memory_context:
                messages.insert(1, {"role": "system", "content": memory_context})
        self._run_states[run_id] = _RunState(messages=messages, conversation_id=conversation_id)
        self._append_message(run_id, {"role": "user", "content": user_message})
        if should_plan(user_message):
            self.trace_collector.record_event(run_id, "planning_started", {
                "request_length": len(user_message),
            })
            try:
                proposal = await self.plan_planner.propose(
                    user_message,
                    self.registry,
                    lambda plan_messages: self._request_llm(
                        plan_messages, run_id, include_tools=False
                    ),
                    context_messages=messages[1:],
                )
                plan = self.plan_validator.validate(proposal, run_id, user_message)
            except Exception as error:
                if "proposal" in locals():
                    rejected_steps = [PlanStep(**step.model_dump()) for step in proposal.steps]
                    rejected = ExecutionPlan(
                        run_id=run_id, original_request=user_message, steps=rejected_steps,
                        status=PlanStatus.REJECTED,
                        error=f"{type(error).__name__}: {str(error)[:250]}",
                    )
                    self.trace_collector.get_run(run_id).execution_plan = rejected
                    self.trace_collector.record_event(run_id, "plan_rejected", {
                        "plan_id": rejected.plan_id, "error_type": type(error).__name__,
                        "reason": str(error)[:250],
                    })
                else:
                    self.trace_collector.record_event(run_id, "plan_rejected", {
                        "reason": "Planning request failed or returned an invalid schema.",
                        "error_type": type(error).__name__,
                    })
                self.trace_collector.record_event(run_id, "planning_fallback_to_agent_loop")
                return await self._drive(run_id)

            self.trace_collector.get_run(run_id).execution_plan = plan
            self._plan_states[run_id] = _PlanRunState(
                plan=plan, outputs={}, preference_context=memory_context
            )
            self.trace_collector.record_event(run_id, "plan_created", {
                "plan_id": plan.plan_id, "step_count": len(plan.steps),
                "step_ids": plan.execution_order,
            })
            self.trace_collector.record_event(run_id, "plan_validated", {
                "plan_id": plan.plan_id, "step_count": len(plan.steps),
            })
            return await self._advance_plan(run_id)
        return await self._drive(run_id)

    def _append_message(self, run_id: str, message: dict) -> None:
        state = self._run_states[run_id]
        state.messages.append(message)
        if self.conversation_service is not None and state.conversation_id:
            self.conversation_service.append(state.conversation_id, message)

    async def _drive(self, run_id: str):
        state = self._run_states[run_id]
        try:
            while True:
                if state.tool_calls is not None:
                    waiting = await self._continue_tool_calls(run_id, state)
                    if waiting:
                        return self.trace_collector.get_run(run_id)
                    state.tool_calls = None
                    state.next_tool_index = 0
                    state.tool_rounds += 1

                self.trace_collector.record_event(
                    run_id,
                    "llm_request_started",
                    {"tool_round": state.tool_rounds},
                )

                response = await self._request_llm(state.messages, run_id)

                assistant_message = response.choices[0].message

                self.trace_collector.record_event(
                    run_id,
                    "llm_response_received",
                    {
                        "tool_call_count": len(
                            assistant_message.tool_calls or []
                        ),
                        "finish_reason": response.choices[0].finish_reason,
                    },
                )

                if not assistant_message.tool_calls:
                    safe_response, results = self.guardrails.check_output(
                        assistant_message.content
                    )
                    self._record_guardrail_results(run_id, results)
                    persisted_assistant = assistant_message.model_dump(exclude_none=True)
                    persisted_assistant["content"] = safe_response
                    self._append_message(run_id, persisted_assistant)
                    completed = self.trace_collector.complete_run(
                        run_id,
                        safe_response,
                    )
                    self._run_states.pop(run_id, None)
                    return completed

                self._append_message(run_id, assistant_message.model_dump(exclude_none=True))

                if state.tool_rounds >= self.max_tool_rounds:
                    raise RuntimeError("Maximum tool rounds exceeded")
                state.tool_calls = assistant_message.tool_calls
                state.next_tool_index = 0

        except Exception as error:
            self.trace_collector.fail_run(
                run_id=run_id,
                error=f"{type(error).__name__}: {error}",
            )
            self._run_states.pop(run_id, None)
            raise

    async def _advance_plan(self, run_id: str):
        session = self._plan_states[run_id]
        run = self.trace_collector.get_run(run_id)

        async def execute_step(step, resolved):
            return await self._execute_plan_step(run_id, step, resolved)

        result = await self.plan_engine.advance(
            session.plan,
            session.outputs,
            execute_step,
            lambda event_type, details: self.trace_collector.record_event(run_id, event_type, details),
            active_elapsed_seconds=session.active_elapsed_seconds,
            tool_calls_used=session.tool_calls_used,
        )
        session.active_elapsed_seconds = result.elapsed_seconds
        session.tool_calls_used = result.tool_calls_used
        if result.paused_for_approval:
            run.status = "awaiting_approval"
            return run

        session.plan.completed_at = datetime.now(timezone.utc)
        self._plan_states.pop(run_id, None)
        self._run_states.pop(run_id, None)
        if session.plan.status == PlanStatus.COMPLETED:
            final_step = next(step for step in session.plan.steps if step.step_type == StepType.FINAL)
            final_response = str(final_step.output or "")
            self.trace_collector.record_event(run_id, "plan_completed", {
                "plan_id": session.plan.plan_id,
                "completed_steps": sum(step.status == StepStatus.COMPLETED for step in session.plan.steps),
                "tool_calls_used": session.tool_calls_used,
            })
            return self.trace_collector.complete_run(run_id, final_response)

        self.trace_collector.record_event(run_id, "plan_failed", {
            "plan_id": session.plan.plan_id,
            "failed_steps": [step.step_id for step in session.plan.steps
                             if step.status in (StepStatus.FAILED, StepStatus.BLOCKED)],
            "tool_calls_used": session.tool_calls_used,
        })
        return self.trace_collector.fail_run(run_id, session.plan.error or "Execution plan failed.")

    async def _execute_plan_step(self, run_id: str, step: PlanStep, resolved: dict):
        session = self._plan_states[run_id]
        if step.step_type == StepType.TOOL:
            tool = self.registry.get(step.tool_name or "")
            arguments_json = json.dumps(resolved["arguments"])
            try:
                validated, results = self.guardrails.validate_tool_arguments(tool, arguments_json)
            except GuardrailBlocked as blocked:
                self._record_guardrail_results(run_id, [blocked.result])
                self._ensure_plan_tool_call_message(run_id, step, resolved["arguments"])
                self.trace_collector.create_tool_call(run_id, step.tool_call_id, tool.name, resolved["arguments"])
                self.trace_collector.fail_tool_call(run_id, step.tool_call_id, "Tool arguments rejected by guardrails")
                self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                              "content": json.dumps({"error": "Tool arguments rejected by guardrails"})})
                raise
            self._record_guardrail_results(run_id, results)
            self._ensure_plan_tool_call_message(run_id, step, resolved["arguments"])
            self.trace_collector.record_event(run_id, "permission_evaluated", {
                "plan_id": session.plan.plan_id, "step_id": step.step_id,
                "tool_call_id": step.tool_call_id, "tool_name": tool.name,
                "policy": tool.permission_policy.value,
            })
            if tool.permission_policy == PermissionPolicy.DENIED:
                self.trace_collector.create_tool_call(run_id, step.tool_call_id, tool.name, validated.model_dump(mode="json"))
                self.trace_collector.fail_tool_call(run_id, step.tool_call_id,
                                                    "Tool execution is denied by application policy")
                self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                              "content": json.dumps({"error": "Tool execution denied by application policy"})})
                self.trace_collector.record_event(run_id, "tool_execution_denied", {
                    "plan_id": session.plan.plan_id, "step_id": step.step_id,
                })
                raise PermissionError("Tool execution is denied by application policy.")

            if tool.permission_policy == PermissionPolicy.REQUIRES_APPROVAL:
                decision = session.approval_decision if session.pending_approval_id == step.approval_id else None
                if not decision:
                    request = self.approval_store.create(
                        run_id, step.tool_call_id, tool.name, validated.model_dump(mode="json")
                    )
                    step.approval_id = request.approval_id
                    session.pending_approval_id = request.approval_id
                    trace = self.trace_collector.create_tool_call(
                        run_id, step.tool_call_id, tool.name, validated.model_dump(mode="json")
                    )
                    trace.status = "awaiting_approval"
                    trace.approval_id = request.approval_id
                    self.trace_collector.record_event(run_id, "approval_requested", {
                        "approval_id": request.approval_id, "tool_call_id": step.tool_call_id,
                        "tool_name": tool.name, "plan_id": session.plan.plan_id,
                        "step_id": step.step_id,
                    })
                    return StepExecutionResult(paused_for_approval=True)
                session.approval_decision = None
                session.pending_approval_id = None
                if decision == "rejected":
                    self.trace_collector.fail_tool_call(run_id, step.tool_call_id, "Tool execution rejected by user")
                    self.trace_collector.record_event(run_id, "approval_rejected", {
                        "approval_id": step.approval_id, "tool_call_id": step.tool_call_id,
                        "plan_id": session.plan.plan_id, "step_id": step.step_id,
                    })
                    self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                                  "content": json.dumps({"error": "Tool call rejected by user"})})
                    raise PermissionError("Tool execution rejected by user.")
                self.trace_collector.record_event(run_id, "approval_granted", {
                    "approval_id": step.approval_id, "tool_call_id": step.tool_call_id,
                    "plan_id": session.plan.plan_id, "step_id": step.step_id,
                })
                self.trace_collector.record_event(run_id, "tool_execution_after_approval", {
                    "approval_id": step.approval_id, "tool_call_id": step.tool_call_id,
                    "plan_id": session.plan.plan_id, "step_id": step.step_id,
                })
            else:
                self.trace_collector.record_event(run_id, "tool_execution_authorized", {
                    "tool_call_id": step.tool_call_id, "policy": "auto",
                    "plan_id": session.plan.plan_id, "step_id": step.step_id,
                })

            try:
                existing = self.trace_collector.get_tool_call(run_id, step.tool_call_id)
            except KeyError:
                self.trace_collector.create_tool_call(run_id, step.tool_call_id, tool.name,
                                                      validated.model_dump(mode="json"))
            self.trace_collector.mark_tool_call_started(run_id, step.tool_call_id)
            attempt = 0
            while True:
                try:
                    output = await self.executor.execute_validated(tool.name, validated)
                    break
                except Exception as error:
                    transient = isinstance(error, (ToolExecutionTimeout, ConnectionError, OSError))
                    if tool.retry_safe and transient and attempt < self.plan_limits.tool_retry_limit:
                        attempt += 1
                        self.trace_collector.record_event(run_id, "tool_retry_scheduled", {
                            "plan_id": session.plan.plan_id, "step_id": step.step_id,
                            "tool_call_id": step.tool_call_id, "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                        })
                        continue
                    self.trace_collector.fail_tool_call(
                        run_id, step.tool_call_id,
                        f"{type(error).__name__}: planned tool execution failed",
                    )
                    self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                                  "content": json.dumps({"error": "Tool execution failed."})})
                    return StepExecutionResult(executed_tool=True, error=error)
            output_bytes = len(json.dumps(output, default=str).encode("utf-8"))
            if output_bytes > self.plan_limits.max_step_output_bytes:
                self.trace_collector.complete_tool_call(run_id, step.tool_call_id, {
                    "error": "Tool output exceeded the configured plan limit.",
                    "output_bytes": output_bytes,
                })
                self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                              "content": json.dumps({"error": "Tool output exceeded the plan limit."})})
                return StepExecutionResult(
                    executed_tool=True,
                    error=ValueError("Tool output exceeds the configured plan limit."),
                )
            self.trace_collector.complete_tool_call(run_id, step.tool_call_id, output)
            self._append_message(run_id, {"role": "tool", "tool_call_id": step.tool_call_id,
                                          "content": json.dumps(output, default=str)})
            return StepExecutionResult(output=output, executed_tool=True)

        inputs = resolved["inputs"]
        prompt = {
            "request": session.plan.original_request,
            "step": step.description,
            "available_inputs": inputs,
            "saved_preferences": session.preference_context,
            "preference_rule": "The current request overrides any saved preference.",
        }
        messages = [
            {"role": "system", "content": (
                "You are completing one validated execution-plan step. Use only the request "
                "and supplied step outputs. Keep internal reasoning concise."
            )},
            {"role": "user", "content": json.dumps(prompt, default=str)},
        ]
        response = await self._request_llm(messages, run_id, include_tools=False)
        content = response.choices[0].message.content or ""
        if step.step_type == StepType.FINAL:
            safe_response, results = self.guardrails.check_output(content)
            self._record_guardrail_results(run_id, results)
            self._append_message(run_id, {"role": "assistant", "content": safe_response})
            return StepExecutionResult(output=safe_response)
        if not content.strip():
            raise ValueError("Reasoning step returned empty output.")
        return StepExecutionResult(output=content)

    def _ensure_plan_tool_call_message(self, run_id: str, step: PlanStep, arguments: dict) -> None:
        if step.tool_call_logged:
            return
        self._append_message(run_id, {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": step.tool_call_id, "type": "function",
                "function": {"name": step.tool_name, "arguments": json.dumps(arguments)},
            }],
        })
        step.tool_call_logged = True
        session = self._plan_states[run_id]
        self.trace_collector.record_event(run_id, "tool_call_requested", {
            "plan_id": session.plan.plan_id,
            "step_id": step.step_id,
            "tool_call_id": step.tool_call_id,
            "tool_name": step.tool_name,
        })

    async def _continue_tool_calls(self, run_id: str, state: _RunState) -> bool:
        while state.next_tool_index < len(state.tool_calls):
            call = state.tool_calls[state.next_tool_index]
            if not any(t.tool_call_id == call.id for t in self.trace_collector.get_run(run_id).tool_calls):
                self.trace_collector.record_event(run_id, "tool_call_requested", {
                    "tool_call_id": call.id, "tool_name": call.function.name,
                })
            try:
                tool = self.registry.get(call.function.name)
                validated, results = self.guardrails.validate_tool_arguments(
                    tool, call.function.arguments
                )
            except KeyError:
                result = GuardrailResult(
                    passed=False,
                    guardrail="registered_tool",
                    severity="error",
                    stage=GuardrailStage.TOOL_ARGUMENTS,
                    reason="Requested tool is not registered.",
                )
                self._record_guardrail_results(run_id, [result])
                self.trace_collector.create_tool_call(run_id, call.id, call.function.name, {})
                self.trace_collector.fail_tool_call(run_id, call.id, result.reason)
                self._append_message(run_id, {"role": "tool", "tool_call_id": call.id,
                                               "content": json.dumps({"error": result.reason})})
                state.next_tool_index += 1
                continue
            except GuardrailBlocked as blocked:
                self._record_guardrail_results(run_id, [blocked.result])
                self.trace_collector.create_tool_call(run_id, call.id, call.function.name, {})
                self.trace_collector.fail_tool_call(run_id, call.id, blocked.result.reason or "Tool arguments rejected")
                self._append_message(run_id, {"role": "tool", "tool_call_id": call.id,
                                               "content": json.dumps({"error": "Tool arguments rejected by guardrails"})})
                state.next_tool_index += 1
                continue

            self._record_guardrail_results(run_id, results)
            arguments = validated.model_dump(mode="json")
            self.trace_collector.record_event(run_id, "permission_evaluated", {
                "tool_call_id": call.id, "tool_name": call.function.name,
                "policy": tool.permission_policy.value,
            })
            if tool.permission_policy == PermissionPolicy.DENIED:
                self.trace_collector.create_tool_call(run_id, call.id, call.function.name, arguments)
                self.trace_collector.fail_tool_call(run_id, call.id, "Tool execution is denied by application policy")
                self._append_message(run_id, {"role": "tool", "tool_call_id": call.id,
                                               "content": json.dumps({"error": "Tool execution denied by application policy"})})
                self.trace_collector.record_event(run_id, "tool_execution_denied", {"tool_call_id": call.id})
                state.next_tool_index += 1
                continue

            approval_decision = state.decision if state.approval_id else None
            if tool.permission_policy == PermissionPolicy.REQUIRES_APPROVAL and not approval_decision:
                request = self.approval_store.create(run_id, call.id, call.function.name, arguments)
                trace = self.trace_collector.create_tool_call(run_id, call.id, call.function.name, arguments)
                trace.status = "awaiting_approval"
                trace.approval_id = request.approval_id
                state.approval_id = request.approval_id
                state.decision = None
                self.trace_collector.record_event(run_id, "approval_requested", {
                    "approval_id": request.approval_id, "tool_call_id": call.id,
                    "tool_name": call.function.name,
                })
                self.trace_collector.get_run(run_id).status = "awaiting_approval"
                return True

            if tool.permission_policy == PermissionPolicy.REQUIRES_APPROVAL:
                decision = state.decision
                approval_id = state.approval_id
                state.decision = None
                state.approval_id = None
                if decision == "rejected":
                    self.trace_collector.fail_tool_call(run_id, call.id, "Tool execution rejected by user")
                    self.trace_collector.record_event(run_id, "approval_rejected", {"approval_id": approval_id, "tool_call_id": call.id})
                    self._append_message(run_id, {"role": "tool", "tool_call_id": call.id,
                                                   "content": json.dumps({"error": "Tool call rejected by user"})})
                    state.next_tool_index += 1
                    continue
                self.trace_collector.record_event(run_id, "approval_granted", {"approval_id": approval_id, "tool_call_id": call.id})
                self.trace_collector.record_event(run_id, "tool_execution_after_approval", {"approval_id": approval_id, "tool_call_id": call.id})
            else:
                self.trace_collector.record_event(run_id, "tool_execution_authorized", {"tool_call_id": call.id, "policy": "auto"})

            if not any(t.tool_call_id == call.id for t in self.trace_collector.get_run(run_id).tool_calls):
                self.trace_collector.start_tool_call(run_id, call.id, call.function.name, arguments)
            else:
                # An approved call transitions from waiting to execution here.
                self.trace_collector.mark_tool_call_started(run_id, call.id)
            try:
                result = await self.executor.execute_validated(call.function.name, validated)
            except Exception as error:
                self.trace_collector.fail_tool_call(run_id, call.id, str(error))
                # Pair persisted assistant tool calls with a safe tool response so a later
                # conversation turn never receives an orphaned call or raw exception details.
                failed_tool_message = {
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps({"error": "Tool execution failed."}),
                }
                # Persist a matching, sanitized outcome for future turns, but do not
                # mutate this run's LLM context after execution failed or timed out.
                if self.conversation_service is not None and state.conversation_id:
                    self.conversation_service.append(state.conversation_id, failed_tool_message)
                raise
            self.trace_collector.complete_tool_call(run_id, call.id, result)
            self._append_message(run_id, {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            state.next_tool_index += 1
        return False

    def _record_guardrail_results(self, run_id: str, results: list[GuardrailResult]) -> None:
        for result in results:
            prefix = {
                GuardrailStage.INPUT: "input",
                GuardrailStage.TOOL_ARGUMENTS: "tool",
                GuardrailStage.OUTPUT: "output",
            }[result.stage]
            self.trace_collector.record_event(
                run_id,
                f"{prefix}_guardrail_{'passed' if result.passed else 'failed'}",
                {
                    "guardrail": result.guardrail,
                    "severity": result.severity.value,
                    **({"reason": result.reason} if result.reason else {}),
                },
            )

    async def resolve_approval(self, approval_id: str, decision: str):
        if decision not in ("approved", "rejected"):
            raise ValueError("Invalid approval decision")
        request = self.approval_store.get(approval_id)
        if request.status == "expired":
            raise ApprovalConflict("Approval has expired")
        plan_state = self._plan_states.get(request.run_id)
        if plan_state is not None:
            if plan_state.pending_approval_id != approval_id:
                raise ApprovalConflict("Approval is no longer active")
            self.approval_store.resolve(approval_id, decision)
            plan_state.approval_decision = decision
            self.trace_collector.get_run(request.run_id).status = "running"
            self.trace_collector.record_event(request.run_id, "approval_decision_recorded", {
                "approval_id": approval_id, "decision": decision,
                "plan_id": plan_state.plan.plan_id,
            })
            self.trace_collector.record_event(request.run_id, "plan_resumed", {
                "plan_id": plan_state.plan.plan_id,
                "step_id": next(step.step_id for step in plan_state.plan.steps
                                if step.approval_id == approval_id),
            })
            return await self._advance_plan(request.run_id)
        state = self._run_states.get(request.run_id)
        if state is None or state.approval_id != approval_id:
            raise ApprovalConflict("Approval is no longer active")
        # Synchronous transition and claim happen before any await, preventing duplicate execution.
        self.approval_store.resolve(approval_id, decision)
        state.decision = decision
        self.trace_collector.get_run(request.run_id).status = "running"
        self.trace_collector.record_event(request.run_id, "approval_decision_recorded", {
            "approval_id": approval_id, "decision": decision,
        })
        return await self._drive(request.run_id)

    
    async def _request_llm(self, messages: list, run_id: str, include_tools: bool = True):
        for attempt in range(self.max_llm_retries + 1):
            self.trace_collector.record_event(
                run_id,
                "llm_request_attempt_started",
                {"attempt": attempt + 1},
            )

            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.get_openai_tools() if include_tools else [],
                )

            except (APIConnectionError, APITimeoutError) as exc:
                error = exc
                retryable = True

            except APIStatusError as exc:
                error = exc
                retryable = (
                    exc.status_code in (408, 429)
                    or exc.status_code >= 500
                )

            if not retryable or attempt == self.max_llm_retries:
                raise error

            delay = min(2 ** attempt, 8)

            self.trace_collector.record_event(
                run_id,
                "llm_retry_scheduled",
                {
                    "attempt": attempt + 1,
                    "delay_seconds": delay,
                    "error_type": type(error).__name__,
                },
            )

            await asyncio.sleep(delay)
