from __future__ import annotations

"""Run the preregistered strict request-level experiment matrix."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from src.strict_config import load_deployment
from src.strict_simulate import StrictScenario, simulate
from src.strict_trace import load_request_trace


METHOD_PRIMARY = ("scd", "compute_first", "energy_first", "dpp_matched", "round_robin")
METHOD_CORE = ("scd", "compute_first", "dpp_matched")


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def execution_source_hashes(
    runner: Path, dependencies: tuple[Path, ...] = ()
) -> dict[str, str]:
    source_root = Path(__file__).parent
    paths = {
        source_root / "scheduler_core.py",
        source_root / "strict_config.py",
        source_root / "strict_simulate.py",
        source_root / "strict_trace.py",
        source_root / "run_strict_campaign.py",
        runner.resolve(),
        *(path.resolve() for path in dependencies),
    }
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths, key=lambda item: item.name)
    }


def runtime_environment() -> dict[str, str]:
    packages = ("numpy", "numba", "pandas", "scipy")
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **{
            package: importlib.metadata.version(package)
            for package in packages
        },
    }


def add_task(tasks: list[dict[str, Any]], trace: Path, method: str, scenario: StrictScenario, family: str) -> None:
    record = {
        "trace": trace.as_posix(),
        "method": method,
        "scenario": asdict(scenario),
        "family": family,
    }
    record["key"] = canonical_hash(record)[:24]
    if all(task["key"] != record["key"] for task in tasks):
        tasks.append(record)


def formal_tasks(trace_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed in range(20260914, 20260919):
        trace = trace_root / f"binomial_p0004_seed{seed}"
        base = dict(seed=seed, start_day=10, end_day_exclusive=14)
        for method in METHOD_PRIMARY:
            add_task(tasks, trace, method, StrictScenario(name="primary", **base), "primary")
        for ttl in (30.0, 900.0):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(name=f"ttl_{int(ttl)}", cache_ttl_s=ttl, **base),
                    "cache_ttl",
                )
        for multiplier in (0.75, 1.25):
            for method in ("scd", "dpp_matched"):
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"cache_memory_{multiplier:g}",
                        cache_footprint_multiplier=multiplier,
                        **base,
                    ),
                    "cache_memory",
                )
        for method in METHOD_CORE:
            add_task(
                tasks,
                trace,
                method,
                StrictScenario(
                    name="cache_identity_stratum",
                    cache_identity_mode="stratum",
                    **base,
                ),
                "cache_identity",
            )
        for slots in (1, 2, 4, 8, 16):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(name=f"cache_slots_{slots}", cache_slot_cap=slots, **base),
                    "cache_capacity",
                )
        for multiplier in (6.0, 12.0):
            for method in (
                "scd",
                "scd_no_network",
                "scd_independent_network",
                "dpp_matched",
                "compute_first",
            ):
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"network_{multiplier:g}",
                        background_multiplier=multiplier,
                        **base,
                    ),
                    "network",
                )
        for rate_mbps in (1000.0, 100.0):
            for method in (
                "scd",
                "scd_no_network",
                "dpp_matched",
                "compute_first",
            ):
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"two_switch_{int(rate_mbps)}mbps",
                        inter_switch_rate_mbps=rate_mbps,
                        **base,
                    ),
                    "multi_switch",
                )
        for multiplier in (0.25, 4.0):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"payload_{multiplier:g}",
                        payload_multiplier=multiplier,
                        **base,
                    ),
                    "payload",
                )
        for coefficient in (0.0, 0.40):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"power_cv_{coefficient:g}",
                        power_coefficient_of_variation=coefficient,
                        **base,
                    ),
                    "power_uncertainty",
                )
        add_task(
            tasks,
            trace,
            "scd",
            StrictScenario(name="power_margin_removed", power_margin_enabled=False, **base),
            "power_uncertainty",
        )
        for multiplier in (0.5, 1.5):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"thermal_gain_{multiplier:g}",
                        thermal_gain_multiplier=multiplier,
                        **base,
                    ),
                    "thermal_model",
                )
        for multiplier in (64, 256):
            attack_base = dict(seed=seed, start_day=13, end_day_exclusive=14)
            for method in ("scd", "compute_first"):
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"attack_{multiplier}_guarded",
                        attack_multiplier=multiplier,
                        **attack_base,
                    ),
                    "request_flood",
                )
            add_task(
                tasks,
                trace,
                "scd",
                StrictScenario(
                    name=f"attack_{multiplier}_unguarded",
                    attack_multiplier=multiplier,
                    admission_guard=False,
                    **attack_base,
                ),
                "request_flood",
            )
        for factor in (0.30, 0.10):
            for method in (
                "scd",
                "scd_no_debt",
                "scd_shared_energy",
                "dpp_matched",
                "compute_first",
            ):
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"energy_{factor:g}", energy_budget_factor=factor, **base
                    ),
                    "energy",
                )
        for multiplier in (0.5, 2.0):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"deadline_{multiplier:g}",
                        deadline_floor_multiplier=multiplier,
                        deadline_tail_multiplier_factor=multiplier,
                        **base,
                    ),
                    "deadline_definition",
                )
        for service_variability in (0.5, 1.0, 1.5):
            for startup_variability in (0.5, 1.0, 1.5):
                if service_variability == 1.0 and startup_variability == 1.0:
                    continue
                add_task(
                    tasks,
                    trace,
                    "scd",
                    StrictScenario(
                        name=f"variability_s{service_variability:g}_c{startup_variability:g}",
                        service_variability_multiplier=service_variability,
                        startup_variability_multiplier=startup_variability,
                        **base,
                    ),
                    "variability",
                )
        for profile in ("batch1", "batch64"):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"service_profile_{profile}",
                        service_rate_profile=profile,
                        **base,
                    ),
                    "service_calibration",
                )
        for fraction in (0.10, 0.30):
            for method in METHOD_CORE:
                add_task(
                    tasks,
                    trace,
                    method,
                    StrictScenario(
                        name=f"accelerator_{fraction:g}",
                        accelerator_fraction=fraction,
                        **base,
                    ),
                    "compatibility",
                )
        for method in ("scd_no_thermal_term", "scd_no_forecast", "scd_static_weights"):
            add_task(tasks, trace, method, StrictScenario(name=method, **base), "ablation")
        for method in ("scd", "compute_first"):
            add_task(
                tasks,
                trace,
                method,
                StrictScenario(
                    name="port4_sensitivity", include_sensitivity_nodes=True, **base
                ),
                "topology",
            )

    for seed in range(20260914, 20260919):
        trace = trace_root / f"unbiased_p0004_seed{seed}"
        for method in METHOD_CORE:
            add_task(
                tasks,
                trace,
                method,
                StrictScenario(
                    name="unbiased_rounding",
                    seed=seed,
                    start_day=10,
                    end_day_exclusive=14,
                ),
                "sampling",
            )
    trace = trace_root / "deterministic_p0004"
    for method in METHOD_CORE:
        add_task(
            tasks,
            trace,
            method,
            StrictScenario(
                name="deterministic_rounding",
                seed=20260914,
                start_day=10,
                end_day_exclusive=14,
            ),
            "sampling",
        )
    for seed in range(20260914, 20260919):
        trace = trace_root / f"binomial_stratified_p0004_seed{seed}"
        for method in METHOD_CORE:
            add_task(
                tasks,
                trace,
                method,
                StrictScenario(
                    name="stratified_within_minute",
                    seed=seed,
                    start_day=10,
                    end_day_exclusive=14,
                ),
                "arrival_reconstruction",
            )
    tasks.sort(key=lambda item: (item["trace"], item["family"], item["scenario"]["name"], item["method"]))
    return tasks


def _run_trace_group(
    tasks: list[dict[str, Any]],
    deployment_path: str,
    output_directory: str,
) -> list[dict[str, Any]]:
    output_root = Path(output_directory)
    trace = load_request_trace(Path(tasks[0]["trace"]), verify_hashes=True)
    primary = load_deployment(Path(deployment_path))
    sensitivity = load_deployment(Path(deployment_path), include_sensitivity_nodes=True)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        result_path = output_root / "runs" / f"{task['key']}.json"
        if result_path.exists():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            expected_simulator = hashlib.sha256(
                Path(__file__).with_name("strict_simulate.py").read_bytes()
            ).hexdigest()
            expected_core = hashlib.sha256(
                Path(__file__).with_name("scheduler_core.py").read_bytes()
            ).hexdigest()
            expected_forecast = hashlib.sha256(
                primary.service_forecast_path.read_bytes()
            ).hexdigest()
            if (
                record.get("deployment_sha256") == primary.source_sha256
                and record.get("simulator_sha256") == expected_simulator
                and record.get("scheduler_core_sha256") == expected_core
                and record.get("service_forecast_sha256") == expected_forecast
            ):
                rows.append(record)
                continue
            raise ValueError(f"stale result exists for task {task['key']}: {result_path}")
        scenario = StrictScenario(**task["scenario"])
        deployment = sensitivity if scenario.include_sensitivity_nodes else primary
        save_outcome = (
            output_root / "outcomes" / f"{task['key']}.npz"
            if task["family"] == "primary" and task["method"] in ("scd", "compute_first")
            else None
        )
        result = simulate(
            trace,
            deployment,
            task["method"],
            scenario,
            save_outcomes=save_outcome,
        )
        record = {
            "key": task["key"],
            "family": task["family"],
            "trace": task["trace"],
            "scenario_parameters": task["scenario"],
            **asdict(result),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rows.append(record)
    return rows


def run_campaign(
    trace_root: Path,
    deployment_path: Path,
    output_directory: Path,
    workers: int,
) -> list[dict[str, Any]]:
    tasks = formal_tasks(trace_root)
    missing = sorted({task["trace"] for task in tasks if not Path(task["trace"]).exists()})
    if missing:
        raise FileNotFoundError(f"missing request traces: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    audited_deployment = load_deployment(deployment_path)
    protocol = {
        "schema": "tnsm.strict-campaign.v5",
        "tasks": tasks,
        "task_count": len(tasks),
        "paired_test_days": [10, 14],
        "deployment": deployment_path.as_posix(),
        "deployment_sha256": audited_deployment.source_sha256,
        "policy": "fixed manuscript SCD formula in scheduler_core.py",
        "scheduler_core_sha256": hashlib.sha256(
            Path(__file__).with_name("scheduler_core.py").read_bytes()
        ).hexdigest(),
        "simulator_sha256": hashlib.sha256(
            Path(__file__).with_name("strict_simulate.py").read_bytes()
        ).hexdigest(),
        "service_forecast": audited_deployment.service_forecast_path.as_posix(),
        "service_forecast_sha256": hashlib.sha256(
            audited_deployment.service_forecast_path.read_bytes()
        ).hexdigest(),
        "execution_source_sha256": execution_source_hashes(Path(__file__)),
        "runtime_environment": runtime_environment(),
        "state_protocol": "Each run advances all requests from day 0 and scores only the registered held-out interval.",
        "multiplicity_policy": "All registered contrasts are reported with Holm adjustment within each family.",
        "outcome_archiving": "Per-request outcomes retained for primary SCD and compute-first runs.",
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
    encoded = json.dumps(protocol, indent=2)
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("existing campaign protocol differs from the requested protocol")
    protocol_path.write_text(encoded, encoding="utf-8")

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
            flat = pd.json_normalize(rows, sep=".")
            flat.sort_values("key").to_csv(output_directory / "campaign_results.csv", index=False)
    if len(rows) != len(tasks) or len({row["key"] for row in rows}) != len(tasks):
        raise AssertionError("campaign did not return exactly one result per registered task")
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
    rows = run_campaign(
        args.trace_root, args.deployment, args.out_dir, args.workers
    )
    print(json.dumps({"complete": True, "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
