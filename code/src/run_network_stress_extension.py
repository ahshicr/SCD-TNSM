from __future__ import annotations

"""Run the preregistered full-trace network-stress extension.

This module deliberately imports the frozen strict simulator rather than
modifying it.  The three conditions were fixed before inspecting any results.
Every run advances state from day 0 and scores held-out days 10--13.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.run_strict_campaign import (
    _run_trace_group,
    add_task,
    execution_source_hashes,
    runtime_environment,
)
from src.strict_config import load_deployment
from src.strict_simulate import StrictScenario


METHODS = (
    "scd",
    "scd_no_network",
    "scd_independent_network",
    "dpp_matched",
    "compute_first",
)

CONDITIONS = (
    {
        "name": "stress_100mbps_bg12_payload4",
        "inter_switch_rate_mbps": 100.0,
        "background_multiplier": 12.0,
        "payload_multiplier": 4.0,
    },
    {
        "name": "stress_10mbps_bg12_payload4",
        "inter_switch_rate_mbps": 10.0,
        "background_multiplier": 12.0,
        "payload_multiplier": 4.0,
    },
    {
        "name": "stress_10mbps_bg12_payload16",
        "inter_switch_rate_mbps": 10.0,
        "background_multiplier": 12.0,
        "payload_multiplier": 16.0,
    },
)


def formal_tasks(trace_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed in range(20260914, 20260919):
        trace = trace_root / f"binomial_p0004_seed{seed}"
        for condition in CONDITIONS:
            for method in METHODS:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        seed=seed,
                        start_day=10,
                        end_day_exclusive=14,
                        **condition,
                    ),
                    "network_stress_extension",
                )
    tasks.sort(key=lambda item: (item["trace"], item["scenario"]["name"], item["method"]))
    return tasks


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_extension(
    trace_root: Path, deployment_path: Path, output_directory: Path, workers: int
) -> list[dict[str, Any]]:
    tasks = formal_tasks(trace_root)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).exists()})
    if missing:
        raise FileNotFoundError(f"missing request traces: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    deployment = load_deployment(deployment_path)
    protocol = {
        "schema": "tnsm.network-stress-extension.v3",
        "preregistered_before_result_inspection": True,
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "seeds": list(range(20260914, 20260919)),
        "tasks": tasks,
        "task_count": len(tasks),
        "paired_test_days": [10, 14],
        "deployment": deployment_path.as_posix(),
        "deployment_sha256": deployment.source_sha256,
        "simulator_sha256": _file_hash(Path(__file__).with_name("strict_simulate.py")),
        "scheduler_core_sha256": _file_hash(Path(__file__).with_name("scheduler_core.py")),
        "service_forecast": deployment.service_forecast_path.as_posix(),
        "service_forecast_sha256": _file_hash(deployment.service_forecast_path),
        "execution_source_sha256": execution_source_hashes(Path(__file__)),
        "runtime_environment": runtime_environment(),
        "state_protocol": "Every run advances all requests from day 0 and scores only days 10 through 13.",
        "multiplicity_policy": "All registered metric contrasts use Holm adjustment within this extension family.",
        "selection_policy": "All three fixed conditions and all registered methods are reported, irrespective of significance.",
        "metric_boundaries": {
            "task_energy_mj": "dispatch-attributed energy of requests released in the held-out interval, including their drain completions",
            "supplied_energy_mj": "modeled idle plus incremental power integrated on the fixed held-out wall-clock interval",
            "peak_aoi": "receiver-order pre-delivery age for informative updates after each raw-function stream is initialized",
            "success_ratio": "strict-deadline on-time completions divided by all offered requests",
            "deadline_ceiling_success_ratio": "ceiling-deadline on-time completions divided by all offered requests, equal to strict success when no ceiling guard is active",
        },
        "resource_time_semantics": {
            "cross_traffic": "sampled at the actual or predicted service start of each traversed hop",
            "power_factor": "sampled at realized device execution start and held for that request",
            "drain_beyond_trace": "the final available exogenous value is held constant",
        },
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("existing extension protocol differs from the requested protocol")
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
        raise AssertionError("extension did not return one result per registered task")
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
