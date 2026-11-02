import re
from typing import Any


_SECRET_KEY = re.compile(r"(api[_-]?key|secret|password|authorization|access[_-]?token|refresh[_-]?token)", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*|\b(sk|rk)-[A-Za-z0-9_-]{12,}\b|"
    r"((?:api[_ -]?key|access token)\s*[:=]\s*)\S+"
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(child)
                for key, child in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub(lambda match: (match.group(1) or match.group(3) or "") + "[REDACTED]", value)
    return value


def trace_event_rows(trace: dict) -> list[list[str]]:
    rows = []
    for event in trace.get("events", []):
        rows.append([
            str(event.get("timestamp", "")),
            str(event.get("event_type", "")),
            str(redact(event.get("details", {}))),
        ])
    return rows


def tool_call_rows(trace: dict) -> list[list[str]]:
    rows = []
    for call in trace.get("tool_calls", []):
        rows.append([
            str(call.get("tool_name", "")),
            str(call.get("status", "")),
            str(call.get("duration_ms") or ""),
            str(redact(call.get("arguments", {}))),
            str(redact(call.get("result"))) if call.get("result") is not None else "",
            str(redact(call.get("error") or "")),
        ])
    return rows
