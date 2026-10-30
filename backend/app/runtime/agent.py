import json

from openai import AsyncOpenAI

from app.tools.registry import ToolRegistry
from app.tools.executor import ToolExecutor


class AgentRuntime:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        registry: ToolRegistry,
        max_tool_rounds: int = 5,
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.max_tool_rounds = max_tool_rounds

    async def run(self, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are BookRecommender, an AI book assistant. "
                    "Use available tools when they help answer "
                    "the user's request."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        tool_rounds = 0

        while True:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.registry.get_openai_tools(),
            )

            assistant_message = response.choices[0].message

            # Preserve the assistant response, including tool calls.
            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            # No tool calls means the model has finished.
            if not assistant_message.tool_calls:
                return assistant_message.content or ""

            if tool_rounds >= self.max_tool_rounds:
                raise RuntimeError("Maximum tool rounds exceeded")

            # Execute every tool requested in this response.
            for tool_call in assistant_message.tool_calls:
                result = await self.executor.execute(
                    tool_name=tool_call.function.name,
                    arguments_json=tool_call.function.arguments,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            tool_rounds += 1