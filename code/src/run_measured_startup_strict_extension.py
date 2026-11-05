from __future__ import annotations

"""Run the physical-startup sensitivity on the canonical request simulator."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.run_strict_campaign import canonical_hash, execution_source_hashes, runtime_environment
from src.startup_profiles import (
    PROFILE_NAMES,
    canonical_hash as profile_hash,
    deployment_with_startup_profile,
    derive_startup_profiles,
)
from src.strict_config import load_deployment
from src.strict_simulate import StrictScenario, simulate
from src.strict_trace import load_request_trace


METHODS = ("scd", "dpp_matched", "compute_first")
TTLS = (30.0, 60.0, 300.0, 900.0)
SEEDS = tuple(range(20260914, 20260919))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_tasks(trace_root: Path, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed in SEEDS:
        trace = trace_root / f"binomial_p0004_seed{seed}"
        for profile_name in PROFILE_NAMES:
            profile = evidence["profiles"][profile_name]
            profile_sha256 = profile_hash(profile)
            for ttl in TTLS:
                for method in METHODS:
                    scenario = StrictScenario(
                        name=f"{profile_name}_ttl_{int(ttl)}",
                        seed=seed,
                        start_day=10,
                        end_day_exclusive=14,
                        cache_ttl_s=ttl,
                    )
                    record = {
                        "trace": trace.as_posix(),
                        "method": method,
                        "scenario": asdict(scenario),
                        "family": "measured_startup",
                        "startup_profile": profile_name,
                        "startup_profile_sha256": profile_sha256,
                    }
                    record["key"] = canonical_hash(record)[:24]
                    tasks.append(record)
    tasks.sort(
        key=lambda row: (
            row["trace"],
            row["startup_profile"],
            row["scenario"]["cache_ttl_s"],
            row["method"],
        )
    )
    if len(tasks) != 180 or len({row["key"] for row in tasks}) != 180:
        raise AssertionError("measured-startup task matrix is incomplete or duplicated")
    return tasks


def _run_trace_group(
    tasks: list[dict[str, Any]],
    deployment_path: str,
    output_directory: str,
    evidence: dict[str, Any],
    effective_deployment_sha256: str,
) -> list[dict[str, Any]]:
    trace = load_request_trace(Path(tasks[0]["trace"]), verify_hashes=True)
    base_deployment = load_deployment(Path(deployment_path))
    deployments = {
        name: deployment_with_startup_profile(
            base_deployment,
            evidence,
            name,
            effective_deployment_sha256,
        )
        for name in PROFILE_NAMES
    }
    output_root = Path(output_directory)
    expected_simulator = file_hash(Path(__file__).with_name("strict_simulate.py"))
    expected_scheduler = file_hash(Path(__file__).with_name("scheduler_core.py"))
    expected_forecast = file_hash(base_deployment.service_forecast_path)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        result_path = output_root / "runs" / f"{task['key']}.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            expected = {
                "deployment_sha256": effective_deployment_sha256,
                "simulator_sha256": expected_simulator,
                "scheduler_core_sha256": expected_scheduler,
                "service_forecast_sha256": expected_forecast,
                "startup_profile_sha256": task["startup_profile_sha256"],
            }
            if all(record.get(name) == value for name, value in expected.items()):
                rows.append(record)
                continue
            raise ValueError(f"stale result exists for measured-startup task {task['key']}")
        deployment = deployments[task["startup_profile"]]
        result = simulate(
            trace,
            deployment,
            task["method"],
            StrictScenario(**task["scenario"]),
        )
        record = {
            "key": task["key"],
            "family": task["family"],
            "trace": task["trace"],
            "startup_profile": task["startup_profile"],
            "startup_profile_sha256": task["startup_profile_sha256"],
            "scenario_parameters": task["scenario"],
            **asdict(result),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rows.append(record)
    return rows


def run_extension(
    trace_root: Path,
    deployment_path: Path,
    container_summary: Path,
    activation_summary: Path,
    output_directory: Path,
    workers: int,
) -> list[dict[str, Any]]:
    evidence = derive_startup_profiles(container_summary, activation_summary)
    tasks = formal_tasks(trace_root, evidence)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).is_dir()})
    if missing:
        raise FileNotFoundError(f"missing request traces: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    base_deployment = load_deployment(deployment_path)
    effective_deployment_sha256 = canonical_hash(
        {
            "base_deployment_sha256": base_deployment.source_sha256,
            "startup_evidence_sha256": evidence["profile_sha256"],
        }
    )
    protocol = {
        "schema": "tnsm.strict-measured-startup-extension.v1",
        "tasks": tasks,
        "task_count": len(tasks),
        "profiles": list(PROFILE_NAMES),
        "cache_ttl_seconds": list(TTLS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "paired_test_days": [10, 14],
        "deployment": deployment_path.as_posix(),
        "base_deployment_sha256": base_deployment.source_sha256,
        "deployment_sha256": effective_deployment_sha256,
        "startup_profile_evidence": evidence,
        "startup_profile_evidence_sha256": evidence["profile_sha256"],
        "simulator_sha256": file_hash(Path(__file__).with_name("strict_simulate.py")),
        "scheduler_core_sha256": file_hash(Path(__file__).with_name("scheduler_core.py")),
        "service_forecast": base_deployment.service_forecast_path.as_posix(),
        "service_forecast_sha256": file_hash(base_deployment.service_forecast_path),
        "execution_source_sha256": execution_source_hashes(
            Path(__file__), (Path(__file__).with_name("startup_profiles.py"),)
        ),
        "runtime_environment": runtime_environment(),
        "state_protocol": (
            "Every run evolves request, cache, queue, thermal, energy, and forecast state from day 0 "
            "and scores held-out days 10 through 13. Cache expiry is tested at predicted and realized "
            "execution start with finite byte-capacity LRU eviction."
        ),
        "profile_boundary": (
            "Pi startup distributions are direct cached-container measurements. RDK X5 and Jetson "
            "Nano combine measured host activation with Pi container surcharges and remain sensitivity brackets."
        ),
        "multiplicity_policy": (
            "All registered method and profile contrasts are reported with Holm adjustment within this family."
        ),
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing measured-startup protocol differs from the requested protocol")
    else:
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
                evidence,
                effective_deployment_sha256,
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
        raise AssertionError("measured-startup extension did not return every registered task")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument(
        "--deployment", type=Path, default=Path("config/strict_deployment_20260914.json")
    )
    parser.add_argument(
        "--container-summary",
        type=Path,
        default=Path(
            "results/revision/physical/container_startup_full_v6_complete_20260906/analysis/summary.json"
        ),
    )
    parser.add_argument(
        "--activation-summary",
        type=Path,
        default=Path(
            "results/revision/physical/activation_transport_isolated_20260905/activation_transport_analysis.json"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = run_extension(
        args.trace_root,
        args.deployment,
        args.container_summary,
        args.activation_summary,
        args.out_dir,
        args.workers,
    )
    print(json.dumps({"complete": True, "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
