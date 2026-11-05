from __future__ import annotations

"""Run the native-resolution low-load breakdown-point experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.prepare_breakdown_traces import PROBABILITIES, SEEDS, trace_directory
from src.run_strict_campaign import (
    _run_trace_group,
    add_task,
    execution_source_hashes,
    runtime_environment,
)
from src.strict_config import load_deployment
from src.strict_simulate import StrictScenario


METHODS = ("scd", "dpp_matched", "compute_first")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_tasks(trace_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for probability in PROBABILITIES:
        for seed in SEEDS:
            trace = trace_directory(trace_root, probability, seed)
            scenario_name = f"breakdown_p{probability:.0e}".replace("+", "").replace("-", "m")
            for method in METHODS:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=scenario_name,
                        seed=seed,
                        start_day=10,
                        end_day_exclusive=14,
                    ),
                    "breakdown_extension",
                )
    tasks.sort(key=lambda item: (item["trace"], item["method"]))
    return tasks


def run_extension(
    trace_root: Path, deployment_path: Path, output_directory: Path, workers: int
) -> list[dict[str, Any]]:
    tasks = formal_tasks(trace_root)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).exists()})
    if missing:
        raise FileNotFoundError(f"missing registered breakdown traces: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    deployment = load_deployment(deployment_path)
    registry = trace_root / "breakdown_trace_registry.json"
    if not registry.is_file():
        raise FileNotFoundError(f"missing breakdown trace registry: {registry}")
    protocol = {
        "schema": "tnsm.breakdown-extension.v1",
        "reviewer_requested_diagnostic": True,
        "registered_before_low_load_result_inspection": True,
        "definition": (
            "the smallest tested native thinning probability at which at least one strict "
            "deadline miss occurs, reported with the geometric-grid resolution and all seeds"
        ),
        "probabilities": list(PROBABILITIES),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "tasks": tasks,
        "task_count": len(tasks),
        "paired_test_days": [10, 14],
        "deployment": deployment_path.as_posix(),
        "deployment_sha256": deployment.source_sha256,
        "simulator_sha256": sha256(Path(__file__).with_name("strict_simulate.py")),
        "scheduler_core_sha256": sha256(Path(__file__).with_name("scheduler_core.py")),
        "service_forecast": deployment.service_forecast_path.as_posix(),
        "service_forecast_sha256": sha256(deployment.service_forecast_path),
        "trace_registry": registry.as_posix(),
        "trace_registry_sha256": sha256(registry),
        "execution_source_sha256": execution_source_hashes(
            Path(__file__), (Path(__file__).with_name("prepare_breakdown_traces.py"),)
        ),
        "runtime_environment": runtime_environment(),
        "state_protocol": (
            "Every run advances all requests from day 0 and scores only days 10 through 13."
        ),
        "selection_policy": "All probabilities, methods, and seeds are reported.",
        "multiplicity_policy": (
            "The breakdown point is a descriptive capacity diagnostic. Method contrasts use "
            "Holm adjustment within the extension family."
        ),
        "metric_boundaries": {
            "success_ratio": "strict on-time completions divided by all offered requests",
            "miss_ratio": "strict late completions divided by admitted requests",
            "rejections": "every rejection is an offered-service failure",
        },
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("existing breakdown protocol differs from the requested protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(task["trace"], []).append(task)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_trace_group, group, str(deployment_path), str(output_directory)): trace
            for trace, group in groups.items()
        }
        for future in as_completed(futures):
            group_rows = future.result()
            rows.extend(group_rows)
            print(
                json.dumps(
                    {
                        "completed_trace": futures[future],
                        "runs": len(group_rows),
                        "completed_runs": len(rows),
                        "total_runs": len(tasks),
                    }
                ),
                flush=True,
            )
            pd.json_normalize(rows, sep=".").sort_values("key").to_csv(
                output_directory / "campaign_results.csv", index=False
            )
    if len(rows) != len(tasks) or len({row["key"] for row in rows}) != len(tasks):
        raise AssertionError("breakdown extension did not return one result per registered task")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument(
        "--deployment", type=Path, default=Path("config/strict_deployment_20260914.json")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = run_extension(args.trace_root, args.deployment, args.out_dir, args.workers)
    print(json.dumps({"complete": True, "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
