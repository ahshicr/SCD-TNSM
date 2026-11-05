from __future__ import annotations

"""Audit whether the active artifact executes the stated request-level model.

This audit does not certify empirical truth. It verifies that the active code,
configuration, traces, and completed campaign protocols agree on the policy
semantics that are easy to accidentally simplify during a revision.
"""

import argparse
from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from src import scheduler_core
from src.run_deadline_guard_extension import formal_tasks as deadline_tasks
from src.run_breakdown_extension import formal_tasks as breakdown_tasks
from src.run_load_scaling_extension import formal_tasks as load_tasks
from src.run_network_stress_extension import formal_tasks as network_tasks
from src.run_strict_campaign import formal_tasks as campaign_tasks
from src.strict_config import Deployment, load_deployment
from src.strict_simulate import (
    METHODS,
    StrictResult,
    StrictScenario,
    _cache_commit,
    _cache_status,
    _interval_overlap,
    _receiver_aoi,
    _route_preview,
    _simulate_kernel,
    build_background_load,
    build_requirement_mask,
)
from src.strict_trace import NETWORK_RANDOM_DIM, TRACE_SCHEMA, load_request_trace


ROOT = Path(__file__).resolve().parents[1]
RETIRED_DEPLOYMENT_FIELDS = {
    "link_jitter_mu_log_s",
    "link_jitter_sigma_log_s",
    "fairness_debt_scale",
    "fairness_debt_clip",
    "legacy_score_weights",
}
RETIRED_KERNEL_NAMES = {
    "tenant_service_debt",
    "target_success_ratio",
    "fairness_debt_scale",
    "fairness_debt_clip",
    "legacy_score_weights",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_protocol(path: Path) -> tuple[bool, dict[str, Any]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    sources = protocol.get("execution_source_sha256", {})
    mismatches: list[str] = []
    for filename, expected in sorted(sources.items()):
        source = ROOT / "src" / filename
        if not source.is_file():
            mismatches.append(f"missing:{filename}")
        elif file_hash(source) != expected:
            mismatches.append(f"changed:{filename}")
    run_files = sorted((path.parent / "runs").glob("*.json"))
    task_count = int(protocol.get("task_count", -1))
    if len(run_files) != task_count:
        mismatches.append(f"runs:{len(run_files)}!={task_count}")
    return not mismatches, {
        "protocol": str(path.resolve()),
        "registered_tasks": task_count,
        "run_files": len(run_files),
        "registered_sources": sources,
        "mismatches": mismatches,
    }


def run_audit(
    trace_root: Path,
    deployment_path: Path,
    protocols: list[Path],
    exact_report: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    scenario = StrictScenario()
    record(
        "primary cache identity is raw function",
        scenario.cache_identity_mode == "function",
        scenario.cache_identity_mode,
    )
    record(
        "primary cache capacity is memory-derived",
        scenario.cache_slot_cap == 0,
        {"cache_slot_cap": scenario.cache_slot_cap},
    )

    services = np.full(2, -1, dtype=np.int32)
    completions = np.zeros(2, dtype=np.float64)
    memory = np.zeros(2, dtype=np.float32)
    _cache_commit(7, 100.0, 0.0, 1.0, services, completions, memory, 200.0, 10.0)
    cold_at_release = _cache_status(7, 100.0, 9.0, services, completions, 200.0, 10.0)
    cold_at_start = _cache_status(7, 100.0, 20.0, services, completions, 200.0, 10.0)
    record(
        "TTL is evaluated at execution start",
        cold_at_release == 0 and cold_at_start == 1,
        {"status_at_release_9s": int(cold_at_release), "status_at_start_20s": int(cold_at_start)},
    )
    _cache_commit(8, 100.0, 2.0, 3.0, services, completions, memory, 200.0, 100.0)
    result = _cache_commit(9, 100.0, 4.0, 5.0, services, completions, memory, 200.0, 100.0)
    record(
        "byte capacity and LRU eviction are active",
        result == (1, 1) and set(services.tolist()) == {8, 9},
        {"commit_result": list(result), "resident_services": services.tolist()},
    )

    finish, queue, jitter = _route_preview(
        0,
        0.0,
        1.0,
        np.zeros(2),
        np.asarray([[0, 1]], dtype=np.int16),
        np.asarray([2], dtype=np.int8),
        np.asarray([8.0, 8.0]),
        np.zeros(2),
        np.zeros(2),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(2),
        0,
        1.35,
        0.05,
    )
    record(
        "multi-hop store-and-forward serialization is active",
        np.isclose(finish, 2.0301) and np.isclose(queue, 0.0),
        {"finish_s": finish, "queue_s": queue, "jitter_s": jitter},
    )

    finish, queue, jitter = _route_preview(
        0,
        0.0,
        1.0,
        np.asarray([60.0]),
        np.asarray([[0]], dtype=np.int16),
        np.asarray([1], dtype=np.int8),
        np.asarray([8.0]),
        np.zeros(1),
        np.zeros(1),
        np.asarray([[0.0], [0.5]], dtype=np.float32),
        np.zeros(1),
        0,
        1.0,
        0.05,
    )
    record(
        "each hop samples cross traffic at service start",
        np.isclose(finish, 62.35005) and np.isclose(queue, 60.0),
        {"finish_s": finish, "queue_s": queue, "jitter_s": jitter},
    )

    aoi = _receiver_aoi(
        np.asarray([10.0, 6.0, 8.0, 12.0]),
        np.asarray([0.0, 1.0, 3.0, 9.0]),
        np.zeros(4, dtype=np.int32),
        1,
    )
    record(
        "receiver AoI excludes initialization and ignores stale delivery",
        np.isnan(aoi[1]) and np.isclose(aoi[2], 7.0) and np.isnan(aoi[0]) and np.isclose(aoi[3], 9.0),
        {"peaks_s": [None if np.isnan(value) else float(value) for value in aoi]},
    )
    overlap_evidence = [
        _interval_overlap(5.0, 15.0, 10.0, 20.0),
        _interval_overlap(10.0, 20.0, 10.0, 20.0),
        _interval_overlap(0.0, 10.0, 10.0, 20.0),
    ]
    record(
        "supplied-energy activity uses an exact half-open wall-clock interval",
        np.allclose(overlap_evidence, [5.0, 10.0, 0.0]),
        {"overlap_s": overlap_evidence},
    )

    rng = np.random.default_rng(20260914)
    equal = True
    for _ in range(128):
        arguments = (
            rng.uniform(0.0, 10.0),
            rng.uniform(0.1, 4.0),
            rng.uniform(0.01, 10.0),
            0.01,
            rng.uniform(0.0, 3.0),
            rng.uniform(0.0, 3.0),
            rng.uniform(0.001, 3.0),
            rng.uniform(0.0, 3.0),
            rng.uniform(0.0, 3.0),
            rng.uniform(0.25, 10.0),
            rng.uniform(0.0, 4.0),
            rng.uniform(0.0, 4.0),
            rng.uniform(0.0, 4.0),
            rng.uniform(20.0, 80.0),
            65.0,
        )
        equal &= scheduler_core.operating_penalty(*arguments, 5) == scheduler_core.operating_penalty(*arguments, 10)
    record(
        "matched DPP shares the exact SCD score",
        equal and METHODS["dpp_matched"] == 5 and METHODS["scd"] == 10,
        {"comparisons": 128, "difference": "feasibility gate only"},
    )
    score_without_debt = scheduler_core.candidate_score(3.0, 2.0, 4.0, 5.0, 7.0, 6)
    score_with_debt = scheduler_core.candidate_score(3.0, 2.0, 4.0, 5.0, 7.0, 10)
    record(
        "no-debt ablation removes device and tenant score debts",
        np.isclose(score_without_debt, 21.0) and score_with_debt > score_without_debt,
        {"no_debt_score": score_without_debt, "full_scd_score": score_with_debt},
    )

    result_fields = {field.name for field in fields(StrictResult)}
    deadline_fields = {
        "on_time_requests",
        "success_ratio",
        "miss_ratio",
        "deadline_ceiling_on_time_requests",
        "deadline_ceiling_success_ratio",
        "deadline_ceiling_miss_ratio",
        "deadline_guard",
        "deadline_ceiling_multiplier",
    }
    record(
        "strict and ceiling deadline outcomes are reported separately",
        deadline_fields <= result_fields,
        sorted(deadline_fields),
    )
    count_fields = {
        "evaluated_requests",
        "admitted_requests",
        "rejected_requests",
        "on_time_requests",
        "actual_cold_start_requests",
        "predicted_cold_start_requests",
        "energy_underestimated_requests",
        "thermal_derated_requests",
        "peak_aoi_samples",
    }
    record(
        "primary ratios retain integer numerator and denominator provenance",
        count_fields <= result_fields,
        sorted(count_fields),
    )

    deployment = load_deployment(deployment_path)
    record(
        "request admission token cap is exact",
        deployment.admission_cap_per_tenant == 4096 and deployment.admission_window_s == 60.0,
        {
            "requests_per_tenant": deployment.admission_cap_per_tenant,
            "window_s": deployment.admission_window_s,
        },
    )
    record(
        "prediction artifact is past-only prequential",
        "prequential" in deployment.raw["controller"]["service_forecast_evidence"]
        and deployment.service_forecast_path.is_file(),
        {
            "forecast": str(deployment.service_forecast_path.resolve()),
            "sha256": file_hash(deployment.service_forecast_path),
        },
    )

    predicted_background, actual_background = build_background_load(
        deployment, 8, StrictScenario(seed=20260914)
    )
    alpha = deployment.background_forecast_alpha
    causal_background = np.allclose(
        predicted_background[1:],
        (1.0 - alpha) * predicted_background[:-1] + alpha * actual_background[:-1],
    )
    record(
        "background prediction is a one-step causal recurrence",
        causal_background,
        {
            "forecast_rows": int(predicted_background.shape[0]),
            "forecast_uses": "previous forecast and previous realized background only",
        },
    )

    kernel_source = inspect.getsource(_simulate_kernel)
    record(
        "attack labels are absent from the scheduling kernel",
        "is_attack" not in kernel_source and "attacker" not in kernel_source,
        {"reporting_only": "attack labels are joined after scheduling outcomes"},
    )

    deployment_fields = {field.name for field in fields(Deployment)}
    record(
        "retired surrogate parameters are absent from Deployment",
        not (deployment_fields & RETIRED_DEPLOYMENT_FIELDS),
        sorted(deployment_fields & RETIRED_DEPLOYMENT_FIELDS),
    )
    simulator_source = inspect.getsource(__import__("src.strict_simulate", fromlist=["*"]))
    scheduler_source = inspect.getsource(scheduler_core)
    active_source = simulator_source + scheduler_source
    found_retired = sorted(name for name in RETIRED_KERNEL_NAMES if name in active_source)
    record(
        "retired surrogate state is absent from active kernels",
        not found_retired,
        found_retired,
    )
    record(
        "tenant successes are grouped from the canonical per-request deadline outcome",
        "tenant_success = np.bincount(" in simulator_source
        and "admitted_tenant[successes]" in simulator_source,
        "one final request-level Boolean supplies both the global and tenant numerators",
    )
    separate_state_names = {
        "pred_device_available",
        "actual_device_available",
        "pred_link_available",
        "actual_link_available",
        "pred_temperature",
        "actual_temperature",
        "pred_cache_service",
        "actual_cache_service",
    }
    record(
        "predicted and realized resource states are separately represented",
        all(name in kernel_source for name in separate_state_names),
        sorted(name for name in separate_state_names if name not in kernel_source),
    )
    record(
        "realized service uses the empirical duration distribution",
        "_empirical_duration_ms(" in kernel_source
        and "service_z[request]" in kernel_source
        and "actual_service_reference" in kernel_source,
        "per-request common random numbers are mapped through the stored daily quantiles",
    )
    record(
        "thermal feasibility uses the robust predicted power factor and every phase endpoint",
        "predicted_active_w = (" in kernel_source
        and "incremental_w[node] * predicted_power_factor" in kernel_source
        and "temperature_at_start," in kernel_source
        and "temperature_at_service," in kernel_source
        and "temperature_finish," in kernel_source,
        "start, post-startup, and completion temperatures enter the hard candidate screen",
    )
    record(
        "reported peak temperature is clipped to the fixed evaluation interval",
        "_thermal_segment_peak(" in kernel_source
        and "evaluation_start_s," in kernel_source
        and "evaluation_end_s," in kernel_source,
        "constant-power phase extrema are evaluated at clipped interval endpoints",
    )
    record(
        "held-out scoring retains full pre-interval state warm-up and drains completions",
        "request_slice = slice(0, stop_index)" in simulator_source
        and "evaluation_mask = (release_s >= start_s)" in simulator_source
        and "# Drain the remaining completion events" in kernel_source,
        "requests before the scoring interval evolve state and admitted work is drained afterward",
    )
    record(
        "success and admitted-miss denominators are explicit and noninterchangeable",
        "success_ratio=success_count / max(evaluated, 1)" in simulator_source
        and "miss_ratio=1.0 - success_count / max(admitted, 1)" in simulator_source,
        {
            "success_denominator": "all offered requests",
            "miss_denominator": "admitted requests",
        },
    )
    runtime_source_paths = (
        ROOT / "src" / "strict_simulate.py",
        ROOT / "src" / "run_strict_campaign.py",
        ROOT / "src" / "run_network_stress_extension.py",
        ROOT / "src" / "run_deadline_guard_extension.py",
    )
    runtime_source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_source_paths)
    scenario_fields = {field.name for field in fields(StrictScenario)}
    unused_scenario_fields = sorted(
        name for name in scenario_fields if f"scenario.{name}" not in runtime_source
    )
    record(
        "every registered scenario field reaches an active execution or reporting path",
        not unused_scenario_fields,
        unused_scenario_fields,
    )
    deployment_consumers = runtime_source + (ROOT / "src" / "strict_optimal_benchmark.py").read_text(
        encoding="utf-8"
    )
    unused_deployment_fields = sorted(
        name for name in deployment_fields if f"deployment.{name}" not in deployment_consumers
    )
    record(
        "every Deployment field reaches an active execution, audit, or reporting path",
        not unused_deployment_fields,
        unused_deployment_fields,
    )

    expected_task_counts = {
        "strict_main": (len(campaign_tasks(trace_root)), 618),
        "network_stress": (len(network_tasks(trace_root)), 75),
        "deadline_guard": (len(deadline_tasks(trace_root)), 60),
        "load_scaling": (len(load_tasks(trace_root)), 30),
        "breakdown": (len(breakdown_tasks(trace_root)), 165),
    }
    for label, (actual, expected) in expected_task_counts.items():
        record(
            f"{label} matrix is complete and unique",
            actual == expected,
            {"actual": actual, "expected": expected},
        )

    calibration_tasks = campaign_tasks(trace_root)
    calibration_profiles = {
        task["scenario"]["service_rate_profile"]
        for task in calibration_tasks
        if task["family"] == "service_calibration"
    }
    record(
        "request-level results include endpoint service-calibration profile sensitivity",
        calibration_profiles == {"batch1", "batch64"},
        sorted(calibration_profiles),
    )

    trace_manifests: list[dict[str, Any]] = []
    for seed in range(20260914, 20260919):
        manifest_path = trace_root / f"binomial_p0004_seed{seed}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trace_manifests.append(
            {
                "path": str(manifest_path.resolve()),
                "requests": int(manifest["arrays"]["release_s"]["shape"][0]),
                "method": manifest["sampling"]["method"],
                "seed": int(manifest["sampling"]["seed"]),
                "sha256": file_hash(manifest_path),
            }
        )
    record(
        "primary traces use seeded binomial thinning",
        all(item["method"] == "binomial" for item in trace_manifests),
        trace_manifests,
    )

    reference_directory = trace_root / "binomial_p0004_seed20260914"
    reference_trace = load_request_trace(reference_directory, verify_hashes=True)
    source_record = reference_trace.manifest["source"]
    record(
        "the canonical trace preserves all raw function metadata",
        reference_trace.manifest.get("schema") == TRACE_SCHEMA
        and int(source_record["functions"]) == len(reference_trace.memory_mb) == 9000
        and int(reference_trace.function_id.min()) >= 0
        and int(reference_trace.function_id.max()) < 9000,
        {
            "schema": reference_trace.manifest.get("schema"),
            "source_functions": int(source_record["functions"]),
            "metadata_functions": int(len(reference_trace.memory_mb)),
            "observed_function_id_range": [
                int(reference_trace.function_id.min()),
                int(reference_trace.function_id.max()),
            ],
        },
    )
    request_arrays = {
        "release_s",
        "function_id",
        "event_id",
        "service_z",
        "network_z",
        "startup_z",
    }
    manifest_arrays = set(reference_trace.manifest["arrays"])
    retained_requests = int(reference_trace.manifest["sampling"]["retained_requests"])
    record(
        "one retained invocation is one request event without a multiplicity weight",
        reference_trace.requests == retained_requests
        and request_arrays <= manifest_arrays
        and not ({"weight", "multiplicity", "batch_count"} & manifest_arrays),
        {
            "retained_requests": retained_requests,
            "event_rows": reference_trace.requests,
            "forbidden_weight_arrays": sorted(
                {"weight", "multiplicity", "batch_count"} & manifest_arrays
            ),
        },
    )
    release_sample = np.asarray(reference_trace.release_s[::997])
    within_minute = np.mod(release_sample, 60.0)
    record(
        "within-minute release times are continuous rather than minute-boundary batches",
        reference_trace.manifest["sampling"]["release_model"] == "uniform"
        and np.all(within_minute >= 0.0)
        and np.all(within_minute < 60.0)
        and float(np.std(within_minute)) > 10.0,
        {
            "release_model": reference_trace.manifest["sampling"]["release_model"],
            "sampled_offsets": int(within_minute.size),
            "offset_sd_s": float(np.std(within_minute)),
        },
    )
    tenant_consistent = True
    application_tenants: dict[int, int] = {}
    for application, tenant in zip(
        np.asarray(reference_trace.app_id),
        np.asarray(reference_trace.tenant_id),
        strict=True,
    ):
        previous = application_tenants.setdefault(int(application), int(tenant))
        if previous != int(tenant):
            tenant_consistent = False
            break
    record(
        "logical tenant assignment is application coherent",
        tenant_consistent and set(application_tenants.values()) <= {0, 1, 2, 3},
        {
            "applications": len(application_tenants),
            "tenants": sorted(set(application_tenants.values())),
        },
    )
    record(
        "per-request random vectors cover every registered route hop",
        reference_trace.network_z.ndim == 2
        and reference_trace.network_z.shape[1] == NETWORK_RANDOM_DIM
        and NETWORK_RANDOM_DIM >= 3,
        {
            "network_random_dimensions": int(reference_trace.network_z.shape[1]),
            "maximum_registered_hops": 3,
        },
    )
    sampled_indices = np.arange(0, reference_trace.requests, 1009, dtype=np.int64)
    record(
        "service, startup, and network uncertainty are distinct request-level streams",
        sampled_indices.size > 10
        and not np.array_equal(
            reference_trace.service_z[sampled_indices],
            reference_trace.startup_z[sampled_indices],
        )
        and not np.array_equal(
            reference_trace.service_z[sampled_indices],
            reference_trace.network_z[sampled_indices, 0],
        ),
        {
            "sampled_requests": int(sampled_indices.size),
            "stream_names": ["service_z", "startup_z", "network_z"],
        },
    )
    accelerated_requirement = build_requirement_mask(
        reference_trace, StrictScenario(accelerator_fraction=0.25)
    )
    record(
        "capability requirements are function-stable and include accelerator classes",
        len(accelerated_requirement) == 9000
        and np.all((accelerated_requirement & 1) == 1)
        and np.any((accelerated_requirement & 8) == 8)
        and np.any((accelerated_requirement & 16) == 16),
        {
            "portable_cpu_functions": int(np.count_nonzero(accelerated_requirement & 1)),
            "gpu_functions": int(np.count_nonzero(accelerated_requirement & 8)),
            "npu_functions": int(np.count_nonzero(accelerated_requirement & 16)),
        },
    )

    scope_path = ROOT / "IMPLEMENTATION_SCOPE.md"
    scope_text = scope_path.read_text(encoding="utf-8")
    record(
        "physical adapter and canonical simulator evidence are not conflated",
        "Physical evidence boundary" in scope_text
        and "not validate raw-function container identity" in scope_text
        and "must not call them an end-to-end hardware" in scope_text
        and "validation of every request-level mechanism" in scope_text,
        {"scope_file": str(scope_path.resolve()), "sha256": file_hash(scope_path)},
    )

    if exact_report is not None:
        exact = json.loads(exact_report.read_text(encoding="utf-8"))
        expected_rows = int(exact["assignments_per_seed"]) * len(exact["seeds"])
        record(
            "small-instance optimizer is checked against exhaustive state replay",
            exact.get("status") == "PASS"
            and bool(exact.get("exact_dominates_greedy_all_seeds"))
            and expected_rows == 81920,
            {
                "report": str(exact_report.resolve()),
                "report_sha256": file_hash(exact_report),
                "enumerated_assignments": expected_rows,
                "mean_greedy_gap_percent": exact.get("mean_greedy_gap_percent"),
            },
        )

    for protocol_path in protocols:
        passed, evidence = audit_protocol(protocol_path)
        record(f"campaign protocol is source-locked: {protocol_path.parent.name}", passed, evidence)

    failures = [check for check in checks if not check["passed"]]
    return {
        "schema": "tnsm.implementation-conformance.v1",
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "deployment": str(deployment_path.resolve()),
        "deployment_sha256": file_hash(deployment_path),
        "check_results": checks,
        "failed_checks": failures,
        "interpretation": (
            "PASS verifies agreement between active implementation, configuration, trace construction, "
            "and registered campaign sources. It does not relabel transferred calibration values as direct measurements."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument(
        "--deployment", type=Path, default=Path("config/strict_deployment_20260914.json")
    )
    parser.add_argument("--protocol", type=Path, action="append", default=[])
    parser.add_argument("--exact-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        args.trace_root,
        args.deployment,
        args.protocol,
        args.exact_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checks", "passed", "failed")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
