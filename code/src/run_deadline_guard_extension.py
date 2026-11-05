from __future__ import annotations

"""Run a full-trace admission-control diagnostic under network overload.

The diagnostic was added after the fixed network-stress experiment exposed an
unstable best-effort queue at 10 Mb/s.  It reports both strict deadline
rejection and a four-deadline grace ceiling for SCD and compute-first.  The
post-hoc motivation is recorded in the protocol and all registered outcomes
are retained.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.run_network_stress_extension import CONDITIONS
from src.run_strict_campaign import (
    _run_trace_group,
    add_task,
    execution_source_hashes,
    runtime_environment,
)
from src.strict_config import load_deployment
from src.strict_simulate import StrictScenario


MODES = (
    {"label": "strict", "deadline_guard_multiplier": 1.0},
    {"label": "grace4", "deadline_guard_multiplier": 4.0},
)
METHODS = ("scd", "compute_first")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_tasks(trace_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed in range(20260914, 20260919):
        trace = trace_root / f"binomial_p0004_seed{seed}"
        for condition in CONDITIONS:
            for mode in MODES:
                for method in METHODS:
                    add_task(
                        tasks,
                        trace,
                        method,
                        StrictScenario(
                            name=f"{condition['name']}_{mode['label']}",
                            seed=seed,
                            start_day=10,
                            end_day_exclusive=14,
                            deadline_guard=True,
                            deadline_guard_multiplier=mode["deadline_guard_multiplier"],
                            inter_switch_rate_mbps=condition["inter_switch_rate_mbps"],
                            background_multiplier=condition["background_multiplier"],
                            payload_multiplier=condition["payload_multiplier"],
                        ),
                        "deadline_guard_network_stress",
                    )
    tasks.sort(key=lambda item: (item["trace"], item["scenario"]["name"], item["method"]))
    return tasks


def run_extension(
    trace_root: Path, deployment_path: Path, output_directory: Path, workers: int
) -> list[dict[str, Any]]:
    tasks = formal_tasks(trace_root)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).exists()})
    if missing:
        raise FileNotFoundError(f"missing request traces: {missing}")
    deployment = load_deployment(deployment_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "tnsm.deadline-guard-network-stress.v3",
        "design_status": "post-hoc diagnostic registered after observing unbounded best-effort backlog at 10 Mb/s",
        "motivation": "test whether explicit deadline-based rejection prevents work-conserving best-effort overload",
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "methods": list(METHODS),
        "seeds": list(range(20260914, 20260919)),
        "tasks": tasks,
        "task_count": len(tasks),
        "paired_test_days": [10, 14],
        "deployment": deployment_path.as_posix(),
        "deployment_sha256": deployment.source_sha256,
        "simulator_sha256": _hash(Path(__file__).with_name("strict_simulate.py")),
        "scheduler_core_sha256": _hash(Path(__file__).with_name("scheduler_core.py")),
        "service_forecast": deployment.service_forecast_path.as_posix(),
        "service_forecast_sha256": _hash(deployment.service_forecast_path),
        "execution_source_sha256": execution_source_hashes(
            Path(__file__), (Path(__file__).with_name("run_network_stress_extension.py"),)
        ),
        "runtime_environment": runtime_environment(),
        "state_protocol": "Every run advances all requests from day 0 and scores only days 10 through 13.",
        "reporting_policy": "Every registered condition, method and gate mode is reported.",
        "multiplicity_policy": "Holm adjustment across all registered metric contrasts within this diagnostic family.",
        "metric_boundaries": {
            "task_energy_mj": "dispatch-attributed energy of requests released in the held-out interval, including their drain completions",
            "supplied_energy_mj": "modeled idle plus incremental power integrated on the fixed held-out wall-clock interval",
            "peak_aoi": "receiver-order pre-delivery age for informative updates after each raw-function stream is initialized",
            "success_ratio": "strict-deadline on-time completions divided by all offered requests",
            "deadline_ceiling_success_ratio": "ceiling-deadline on-time completions divided by all offered requests",
        },
        "resource_time_semantics": {
            "cross_traffic": "sampled at the actual or predicted service start of each traversed hop",
            "power_factor": "sampled at realized device execution start and held for that request",
            "drain_beyond_trace": "the final available exogenous value is held constant",
        },
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("existing deadline-guard protocol differs from the requested protocol")
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
        raise AssertionError("diagnostic did not return one result per registered task")
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
