from typing import Any


def category_rows(evaluation: dict | None) -> list[list[Any]]:
    if not evaluation:
        return []
    categories = evaluation.get("categories") or []
    return [[item.get("category"), item.get("scenario_count"), item.get("passed_count"),
             item.get("failed_count"), item.get("pass_rate")] for item in categories]


def failed_case_rows(evaluation: dict | None) -> list[list[Any]]:
    if not evaluation:
        return []
    return [[case.get("scenario_id"), case.get("category"), case.get("failure_explanation") or ""]
            for case in evaluation.get("case_results", []) if not case.get("passed")]


def metric_rows(evaluation: dict | None) -> list[list[Any]]:
    if not evaluation:
        return []
    metrics = evaluation.get("metrics") or {}
    return [[key.replace("_", " ").title(), value] for key, value in metrics.items()]
