from typing import Any
from urllib.parse import quote

import httpx


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class BackendAPI:
    """Thin asynchronous HTTP client. It deliberately imports no backend code."""

    def __init__(self, base_url: str, timeout_seconds: float = 90.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(self, method: str, path: str, *, json_body: Any = None) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise APIError("The backend request timed out. Please try again.") from exc
        except httpx.RequestError as exc:
            raise APIError("Cannot reach the backend. Check that the service is running.") from exc

        if response.is_error:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if response.status_code >= 500:
                message = "The backend could not complete the request."
            elif isinstance(detail, str):
                message = detail
            else:
                message = f"The backend rejected the request (HTTP {response.status_code})."
            raise APIError(message, response.status_code)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIError("The backend returned an invalid response.", response.status_code) from exc

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def chat(self, message: str, conversation_id: str | None = None) -> dict:
        body = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return await self._request("POST", "/api/chat", json_body=body)

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        return await self._request("GET", f"/api/conversations?limit={limit}")

    async def get_conversation(self, conversation_id: str) -> dict:
        return await self._request("GET", f"/api/conversations/{quote(conversation_id, safe='')}")

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._request("DELETE", f"/api/conversations/{quote(conversation_id, safe='')}")

    async def get_preferences(self, conversation_id: str) -> list[dict]:
        return await self._request("GET", f"/api/conversations/{quote(conversation_id, safe='')}/preferences")

    async def create_preference(self, conversation_id: str, category: str, value: str) -> dict:
        return await self._request(
            "POST", f"/api/conversations/{quote(conversation_id, safe='')}/preferences",
            json_body={"category": category, "value": value},
        )

    async def delete_preference(self, conversation_id: str, preference_id: str) -> None:
        await self._request(
            "DELETE", f"/api/conversations/{quote(conversation_id, safe='')}/preferences/{quote(preference_id, safe='')}"
        )

    async def clear_preferences(self, conversation_id: str) -> None:
        await self._request("DELETE", f"/api/conversations/{quote(conversation_id, safe='')}/preferences")

    async def list_approvals(self) -> list[dict]:
        return await self._request("GET", "/api/approvals")

    async def get_approval(self, approval_id: str) -> dict:
        return await self._request("GET", f"/api/approvals/{quote(approval_id, safe='')}")

    async def resolve_approval(self, approval_id: str, decision: str) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        action = "approve" if decision == "approve" else "reject"
        return await self._request("POST", f"/api/approvals/{quote(approval_id, safe='')}/{action}")

    async def list_traces(self, limit: int = 30) -> list[dict]:
        return await self._request("GET", f"/api/traces?limit={limit}")

    async def get_trace(self, run_id: str) -> dict:
        return await self._request("GET", f"/api/traces/{quote(run_id, safe='')}")

    async def list_evaluations(self, limit: int = 20) -> list[dict]:
        return await self._request("GET", f"/api/evaluations?limit={limit}")

    async def start_evaluation(self) -> dict:
        return await self._request("POST", "/api/evaluations/run")

    async def get_evaluation(self, evaluation_id: str) -> dict:
        return await self._request("GET", f"/api/evaluations/{quote(evaluation_id, safe='')}")
