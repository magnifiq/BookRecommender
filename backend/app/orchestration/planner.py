import json
import re
from collections.abc import Awaitable, Callable

from app.orchestration.models import PlanProposal
from app.tools.registry import ToolRegistry


LLMInvoker = Callable[[list[dict]], Awaitable[object]]


def should_plan(user_request: str) -> bool:
    """Conservative router: keep ordinary recommendations and lookups lightweight."""
    text = user_request.casefold()
    patterns = (
        r"\b(compare|comparison|rank|evaluate)\b.*\b(and|then|before|after)\b",
        r"\b(find|search|look up)\b.*\b(compare|rank|recommend the best|choose the best)\b",
        r"\b(step[- ]by[- ]step|multi[- ]step|plan this)\b",
        r"\b(research|gather|collect)\b.*\bthen\b.*\b(recommend|summarize|compare)\b",
    )
    return any(re.search(pattern, text, re.DOTALL) for pattern in patterns)


class LLMPlanPlanner:
    """Asks the model for a plan proposal; it has no tool execution capability."""

    async def propose(self, request: str, registry: ToolRegistry, invoke: LLMInvoker,
                      context_messages: list[dict] | None = None) -> PlanProposal:
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "arguments_schema": tool.args_schema.model_json_schema(),
        } for tool in registry.get_all()]
        messages = [
            {"role": "system", "content": (
                "Create a concise execution plan and return only JSON matching "
                "{\"steps\":[{\"step_id\":string,\"description\":string,"
                "\"step_type\":\"tool_execution|llm_reasoning|final_response\","
                "\"depends_on\":[string],\"input_refs\":[string],"
                "\"tool_name\":string|null,\"arguments\":object}]} . "
                "Use only listed tools. For referenced output values use the exact "
                "object {\"$ref\":\"step_id.output.path\"}; list each reference in input_refs "
                "and make its source a direct dependency. Put final response last and "
                "depend on every output needed for the answer. Tool steps execute under application policy; do not "
                "invent permission fields or claim approval."
            )},
            {"role": "user", "content": json.dumps({
                "request": request,
                "recent_conversation_context": context_messages or [],
                "tools": tools,
            })},
        ]
        response = await invoke(messages)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Planner returned an empty plan.")
        return PlanProposal.model_validate_json(content)
