
import json

from openai import AsyncOpenAI

from app.runtime.trace_collector import TraceCollector
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        registry: ToolRegistry,
        trace_collector: TraceCollector,
        max_tool_rounds: int = 5,
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.trace_collector = trace_collector
        self.max_tool_rounds = max_tool_rounds

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

        tool_rounds = 0

        try:
            while True:
                self.trace_collector.record_event(
                    run_id,
                    "llm_request_started",
                    {"tool_round": tool_rounds},
                )

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.get_openai_tools(),
                )

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

                messages.append(
                    assistant_message.model_dump(exclude_none=True)
                )

                if not assistant_message.tool_calls:
                    return self.trace_collector.complete_run(
                        run_id,
                        assistant_message.content or "",
                    )

                if tool_rounds >= self.max_tool_rounds:
                    raise RuntimeError("Maximum tool rounds exceeded")

                for tool_call in assistant_message.tool_calls:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be a JSON object"
                        )

                    self.trace_collector.start_tool_call(
                        run_id=run_id,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        arguments=arguments,
                    )

                    try:
                        result = await self.executor.execute(
                            tool_name=tool_call.function.name,
                            arguments_json=tool_call.function.arguments,
                        )
                    except Exception as tool_error:
                        self.trace_collector.fail_tool_call(
                            run_id=run_id,
                            tool_call_id=tool_call.id,
                            error=str(tool_error),
                        )
                        raise

                    self.trace_collector.complete_tool_call(
                        run_id=run_id,
                        tool_call_id=tool_call.id,
                        result=result,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

                tool_rounds += 1

        except Exception as error:
            self.trace_collector.fail_run(
                run_id=run_id,
                error=f"{type(error).__name__}: {error}",
            )
            raise
