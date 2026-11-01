
import json, asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from app.runtime.trace_collector import TraceCollector
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.base import PermissionPolicy
from app.runtime.approvals import ApprovalStore, ApprovalConflict


@dataclass
class _RunState:
    messages: list
    tool_rounds: int = 0
    tool_calls: list | None = None
    next_tool_index: int = 0
    approval_id: str | None = None
    decision: str | None = None


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
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry, tool_execution_timeout_seconds)
        self.trace_collector = trace_collector
        self.max_tool_rounds = max_tool_rounds
        self.max_llm_retries = max_llm_retries
        self.approval_store = approval_store or ApprovalStore()
        self._run_states: dict[str, _RunState] = {}
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

    async def run(self, user_message: str):
        run = self.trace_collector.start_run(user_message)
        run_id = run.run_id

        messages = [
            {
                "role": "system",
                "content": (
                    "You are BookRecommender, an AI book assistant. "
                    "Use available tools when they help answer "
                    "the user's request."
                ),
            },
            {"role": "user", "content": user_message},
        ]

        self._run_states[run_id] = _RunState(messages=messages)
        return await self._drive(run_id)

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

                state.messages.append(
                    assistant_message.model_dump(exclude_none=True)
                )

                if not assistant_message.tool_calls:
                    completed = self.trace_collector.complete_run(
                        run_id,
                        assistant_message.content or "",
                    )
                    self._run_states.pop(run_id, None)
                    return completed

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

    async def _continue_tool_calls(self, run_id: str, state: _RunState) -> bool:
        while state.next_tool_index < len(state.tool_calls):
            call = state.tool_calls[state.next_tool_index]
            arguments = json.loads(call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            tool = self.registry.get(call.function.name)
            validated = tool.args_schema.model_validate(arguments)
            arguments = validated.model_dump(mode="json")

            self.trace_collector.record_event(run_id, "tool_call_requested", {
                "tool_call_id": call.id, "tool_name": call.function.name,
            })
            self.trace_collector.record_event(run_id, "permission_evaluated", {
                "tool_call_id": call.id, "tool_name": call.function.name,
                "policy": tool.permission_policy.value,
            })
            if tool.permission_policy == PermissionPolicy.DENIED:
                self.trace_collector.create_tool_call(run_id, call.id, call.function.name, arguments)
                self.trace_collector.fail_tool_call(run_id, call.id, "Tool execution is denied by application policy")
                state.messages.append({"role": "tool", "tool_call_id": call.id,
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
                    state.messages.append({"role": "tool", "tool_call_id": call.id,
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
                result = await self.executor.execute(call.function.name, call.function.arguments)
            except Exception as error:
                self.trace_collector.fail_tool_call(run_id, call.id, str(error))
                raise
            self.trace_collector.complete_tool_call(run_id, call.id, result)
            state.messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            state.next_tool_index += 1
        return False

    async def resolve_approval(self, approval_id: str, decision: str):
        if decision not in ("approved", "rejected"):
            raise ValueError("Invalid approval decision")
        request = self.approval_store.get(approval_id)
        if request.status == "expired":
            raise ApprovalConflict("Approval has expired")
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

    
    async def _request_llm(self, messages: list, run_id: str):
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
                    tools=self.registry.get_openai_tools(),
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
