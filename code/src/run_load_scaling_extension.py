from __future__ import annotations

"""Run the registered half, nominal, and double-load request-level comparison."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.prepare_load_scaling_traces import LOADS, SEEDS
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
    for load in LOADS:
        for seed in SEEDS:
            trace = trace_root / f"{load['directory']}_seed{seed}"
            for method in METHODS:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=str(load["label"]),
                        seed=seed,
                        start_day=10,
                        end_day_exclusive=14,
                    ),
                    "load_scaling_extension",
                )
    tasks.sort(key=lambda item: (item["trace"], item["scenario"]["name"], item["method"]))
    return tasks


def run_extension(
    trace_root: Path, deployment_path: Path, output_directory: Path, workers: int
) -> list[dict[str, Any]]:
    tasks = formal_tasks(trace_root)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).exists()})
    if missing:
        raise FileNotFoundError(f"missing registered load traces: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    deployment = load_deployment(deployment_path)
    registry = trace_root / "load_scaling_registry.json"
    if not registry.is_file():
        raise FileNotFoundError(f"missing load trace registry: {registry}")
    protocol = {
        "schema": "tnsm.load-scaling-extension.v1",
        "preregistered_before_result_inspection": True,
        "load_definition": (
            "native function-minute binomial thinning at one half or twice the nominal "
            "probability, followed by one event per retained invocation"
        ),
        "loads": list(LOADS),
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
            Path(__file__), (Path(__file__).with_name("prepare_load_scaling_traces.py"),)
        ),
        "runtime_environment": runtime_environment(),
        "state_protocol": (
            "Every run advances all requests from day 0 and scores only days 10 through 13."
        ),
        "multiplicity_policy": (
            "Holm adjustment across all registered metric contrasts within the load family."
        ),
        "reporting_policy": "Both added loads and all three methods are reported.",
        "metric_boundaries": {
            "task_energy_mj": (
                "dispatch-attributed energy of held-out requests, including drain completions"
            ),
            "supplied_energy_mj": (
                "idle plus incremental power integrated on the fixed held-out wall-clock interval"
            ),
            "success_ratio": "strict on-time completions divided by all offered requests",
            "peak_aoi": (
                "receiver-order pre-delivery age after each raw-function stream is initialized"
            ),
        },
        "resource_time_semantics": {
            "cross_traffic": "sampled at each traversed hop's service start",
            "power_factor": "sampled at realized device execution start",
            "drain_beyond_trace": "the final exogenous value is held constant",
        },
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("existing load-scaling protocol differs from the requested protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(task["trace"], []).append(task)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_trace_group,
                group,
                str(deployment_path),
                str(output_directory),
            ): trace
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
        raise AssertionError("load extension did not return one result per registered task")
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
