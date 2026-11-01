import argparse
import asyncio
from pathlib import Path

from app.evaluation.dataset import load_scenarios
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scripted import make_scripted_runtime


async def run_benchmark():
    scenarios = load_scenarios()
    return await EvaluationRunner(
        scenarios, make_scripted_runtime, auto_resolve_approvals=True
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic agent benchmark.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this file.")
    args = parser.parse_args()
    report = asyncio.run(run_benchmark()).to_json()
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
