import asyncio
from typing import Any
from pydantic import BaseModel

from app.tools.registry import ToolRegistry


class ToolExecutionTimeout(TimeoutError):
    """Raised when a tool handler exceeds its configured time limit."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' exceeded timeout of {timeout_seconds:g} seconds"
        )


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: float = 10.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        tool_name: str,
        arguments_json: str,
    ) -> Any:
        tool = self.registry.get(tool_name)

        arguments = tool.args_schema.model_validate_json(
            arguments_json
        )

        return await self.execute_validated(tool_name, arguments)

    async def execute_validated(self, tool_name: str, arguments: BaseModel) -> Any:
        """Execute an already validated model without parsing untrusted JSON again."""
        tool = self.registry.get(tool_name)

        try:
            return await asyncio.wait_for(
                tool.handler(arguments), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ToolExecutionTimeout(
                tool_name, self.timeout_seconds
            ) from exc
