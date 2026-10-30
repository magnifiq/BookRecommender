from typing import Any

from app.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments_json: str,
    ) -> Any:
        tool = self.registry.get(tool_name)

        arguments = tool.args_schema.model_validate_json(
            arguments_json
        )

        return await tool.handler(arguments)