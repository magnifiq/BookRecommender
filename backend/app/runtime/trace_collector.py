
from datetime import datetime, timezone

from app.runtime.tracing import AgentRun, TraceEvent, ToolCallTrace


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceCollector:
    """Collects and manages in-memory agent execution traces."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    def start_run(self, user_message: str) -> AgentRun:
        run = AgentRun(user_message=user_message)

        self._runs[run.run_id] = run

        self.record_event(
            run.run_id,
            "run_started",
        )

        return run

    def record_event(
        self,
        run_id: str,
        event_type: str,
        details: dict | None = None,
    ) -> TraceEvent:
        run = self._get_run(run_id)

        event = TraceEvent(
            event_type=event_type,
            details=details or {},
        )

        run.events.append(event)
        return event

    def start_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
    ) -> ToolCallTrace:
        run = self._get_run(run_id)

        trace = ToolCallTrace(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        run.tool_calls.append(trace)

        self.record_event(
            run_id,
            "tool_call_started",
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            },
        )

        return trace

    def complete_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        result,
    ) -> ToolCallTrace:
        trace = self._get_tool_call(run_id, tool_call_id)

        trace.status = "completed"
        trace.result = result
        trace.finished_at = utc_now()
        trace.duration_ms = (
            trace.finished_at - trace.started_at
        ).total_seconds() * 1000

        self.record_event(
            run_id,
            "tool_call_completed",
            {
                "tool_call_id": tool_call_id,
                "tool_name": trace.tool_name,
                "duration_ms": trace.duration_ms,
            },
        )

        return trace

    def fail_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        error: str,
    ) -> ToolCallTrace:
        trace = self._get_tool_call(run_id, tool_call_id)

        trace.status = "failed"
        trace.error = error
        trace.finished_at = utc_now()
        trace.duration_ms = (
            trace.finished_at - trace.started_at
        ).total_seconds() * 1000

        self.record_event(
            run_id,
            "tool_call_failed",
            {
                "tool_call_id": tool_call_id,
                "tool_name": trace.tool_name,
                "error": error,
                "duration_ms": trace.duration_ms,
            },
        )

        return trace

    def complete_run(
        self,
        run_id: str,
        final_response: str,
    ) -> AgentRun:
        run = self._get_run(run_id)

        run.status = "completed"
        run.final_response = final_response
        self._finish_run(run)

        self.record_event(run_id, "run_completed")

        return run

    def fail_run(
        self,
        run_id: str,
        error: str,
    ) -> AgentRun:
        run = self._get_run(run_id)

        run.status = "failed"
        run.error = error
        self._finish_run(run)

        self.record_event(
            run_id,
            "run_failed",
            {"error": error},
        )

        return run

    def get_run(self, run_id: str) -> AgentRun:
        return self._get_run(run_id)

    def get_all_runs(self) -> list[AgentRun]:
        return list(self._runs.values())

    def _finish_run(self, run: AgentRun) -> None:
        run.finished_at = utc_now()
        run.duration_ms = (
            run.finished_at - run.started_at
        ).total_seconds() * 1000

    def _get_run(self, run_id: str) -> AgentRun:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run: '{run_id}'")

        return self._runs[run_id]

    def _get_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> ToolCallTrace:
        run = self._get_run(run_id)

        for trace in run.tool_calls:
            if trace.tool_call_id == tool_call_id:
                return trace

        raise KeyError(
            f"Unknown tool call: '{tool_call_id}' "
            f"in run '{run_id}'"
        )
