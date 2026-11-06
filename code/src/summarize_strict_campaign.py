from __future__ import annotations

"""Audit and summarize the formula-faithful strict campaign.

Every inferential row is paired by trace seed. Holm adjustment retains all
registered stochastic contrasts, including zero or undefined differences with
conservative p=1. Single-reconstruction controls have no inferential p value.
The script never
silently drops a failed conservation or provenance check.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "success_ratio",
    "miss_ratio",
    "deadline_ceiling_success_ratio",
    "deadline_ceiling_miss_ratio",
    "mean_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "task_energy_mj",
    "supplied_energy_mj",
    "mean_peak_aoi_ms",
    "p95_peak_aoi_ms",
    "jain_tenant_success",
    "minimum_tenant_success_ratio",
    "actual_cold_start_ratio",
    "predicted_cache_evictions",
    "actual_cache_evictions",
    "energy_underestimation_ratio",
    "energy_shortfall_j",
    "peak_temperature_c",
    "thermal_violations",
    "p95_network_queue_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def mean_ci(values: Iterable[float]) -> tuple[int, float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return 0, math.nan, math.nan, math.nan
    mean = float(array.mean())
    if n == 1:
        return 1, mean, math.nan, math.nan
    half = float(stats.t.ppf(0.975, n - 1) * array.std(ddof=1) / math.sqrt(n))
    return n, mean, mean - half, mean + half


def holm(values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_row(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    family: str,
    contrast: str,
    reference: str,
    challenger: str,
    metric: str,
) -> dict[str, object] | None:
    pair = left[["seed", metric]].merge(
        right[["seed", metric]], on="seed", suffixes=("_reference", "_challenger")
    )
    if pair.empty:
        return None
    difference = (
        pair[f"{metric}_challenger"].to_numpy(dtype=np.float64)
        - pair[f"{metric}_reference"].to_numpy(dtype=np.float64)
    )
    n, mean, lower, upper = mean_ci(difference)
    # Do not make the size of the multiplicity family depend on observed zeros
    # or no-service outcomes. Keep the declared stochastic comparison as p=1
    # when its outcome is undefined. A fixed one-reconstruction control remains
    # descriptive only. Near-constant differences still use the actual t test.
    if len(pair) < 2:
        p_value = math.nan
    elif not np.isfinite(difference).all() or np.all(difference == 0):
        p_value = 1.0
    elif np.std(difference, ddof=1) == 0:
        p_value = 0.0
    else:
        p_value = float(stats.ttest_1samp(difference, 0.0).pvalue)
    return {
        "family": family,
        "contrast": contrast,
        "reference": reference,
        "challenger": challenger,
        "metric": metric,
        "paired_n": n,
        "difference_challenger_minus_reference": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "p_unadjusted": p_value,
    }


def load_runs(directory: Path) -> pd.DataFrame:
    paths = sorted((directory / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no run records in {directory / 'runs'}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    frame = pd.DataFrame(records)
    if frame["key"].duplicated().any():
        raise AssertionError("duplicate campaign keys")
    return frame


def audit(frame: pd.DataFrame, protocol: dict[str, object], directory: Path) -> dict[str, object]:
    failures: list[str] = []
    deployment_path = Path(str(protocol["deployment"]))
    if not deployment_path.is_absolute() and not deployment_path.is_file():
        deployment_path = directory.parents[2] / deployment_path
    deployment_raw = json.loads(deployment_path.read_text(encoding="utf-8"))
    budget_fraction = float(deployment_raw["controller"]["energy_budget_fraction"])
    burst_s = float(deployment_raw["controller"]["energy_burst_minutes"]) * 60.0
    tenant_share = np.asarray(
        deployment_raw["controller"]["tenant_energy_share"], dtype=np.float64
    )
    capacity_by_port = {
        int(node["port"]):
        (float(node["active_w"]) - float(node["idle_w"]))
        * budget_fraction
        * burst_s
        for node in deployment_raw["nodes"]
    }
    if len(frame) != int(protocol["task_count"]):
        failures.append(f"expected {protocol['task_count']} runs, found {len(frame)}")
    for row in frame.itertuples(index=False):
        if row.admitted_requests + row.rejected_requests != row.evaluated_requests:
            failures.append(f"request conservation failed for {row.key}")
        if sum(row.tenant_offered) != row.evaluated_requests:
            failures.append(f"tenant offered count failed for {row.key}")
        if sum(row.tenant_admitted) != row.admitted_requests:
            failures.append(f"tenant admission count failed for {row.key}")
        if sum(row.node_assignments) != row.admitted_requests:
            failures.append(f"node assignment count failed for {row.key}")
        if sum(row.tenant_success) > row.admitted_requests:
            failures.append(f"tenant success count failed for {row.key}")
        if row.on_time_requests != sum(row.tenant_success):
            failures.append(f"on-time count failed for {row.key}")
        if row.deadline_ceiling_on_time_requests < row.on_time_requests:
            failures.append(f"deadline ceiling success count failed for {row.key}")
        if row.deadline_ceiling_on_time_requests > row.admitted_requests:
            failures.append(f"deadline ceiling exceeds admission for {row.key}")
        if not np.isclose(
            row.deadline_ceiling_success_ratio,
            row.deadline_ceiling_on_time_requests / max(row.evaluated_requests, 1),
            rtol=0.0,
            atol=1.0e-15,
        ):
            failures.append(f"deadline ceiling offered ratio failed for {row.key}")
        if not np.isclose(
            row.deadline_ceiling_miss_ratio,
            1.0 - row.deadline_ceiling_on_time_requests / max(row.admitted_requests, 1),
            rtol=0.0,
            atol=1.0e-15,
        ):
            failures.append(f"deadline ceiling admitted ratio failed for {row.key}")
        if row.legitimate_offered_requests != row.evaluated_requests - row.attacker_offered_requests:
            failures.append(f"legitimate offered count failed for {row.key}")
        if row.legitimate_on_time_requests > row.legitimate_offered_requests:
            failures.append(f"legitimate on-time count failed for {row.key}")
        if not np.isclose(
            row.success_ratio,
            row.on_time_requests / max(row.evaluated_requests, 1),
            rtol=0.0,
            atol=1.0e-15,
        ):
            failures.append(f"offered-load success ratio failed for {row.key}")
        if row.actual_cold_start_requests > row.admitted_requests:
            failures.append(f"actual cold-start count failed for {row.key}")
        if row.predicted_cold_start_requests > row.admitted_requests:
            failures.append(f"predicted cold-start count failed for {row.key}")
        if row.false_warm_requests > row.admitted_requests or row.false_cold_requests > row.admitted_requests:
            failures.append(f"cache prediction error count failed for {row.key}")
        if row.thermal_derated_requests > row.admitted_requests:
            failures.append(f"thermal derating count failed for {row.key}")
        if row.energy_underestimated_requests > row.admitted_requests:
            failures.append(f"energy underestimation count failed for {row.key}")
        if row.evaluation_end_s <= row.evaluation_start_s:
            failures.append(f"evaluation boundary failed for {row.key}")
        if not np.isclose(
            row.drain_duration_s,
            max(0.0, row.last_completion_s - row.evaluation_end_s),
            rtol=0.0,
            atol=1.0e-9,
        ):
            failures.append(f"drain duration failed for {row.key}")
        scenario = row.scenario_parameters
        if bool(scenario["energy_guard"]):
            factor = float(scenario["energy_budget_factor"])
            capacities = np.asarray(
                [capacity_by_port[int(port)] * factor for port in row.node_ports],
                dtype=np.float64,
            )
            tolerance = 1.0e-7
            if row.maximum_per_device_energy_debt_j > float(capacities.max()) + tolerance:
                failures.append(f"per-device debt envelope failed for {row.key}")
            if row.final_total_device_energy_debt_j > float(capacities.sum()) + tolerance:
                failures.append(f"total device debt envelope failed for {row.key}")
            tenant_cap = capacities.reshape(-1, 1) * tenant_share.reshape(1, -1)
            if (
                row.maximum_per_device_tenant_energy_debt_j
                > float(tenant_cap.max()) + tolerance
            ):
                failures.append(f"per-device-tenant debt envelope failed for {row.key}")
            if row.final_total_tenant_energy_debt_j > float(tenant_cap.sum()) + tolerance:
                failures.append(f"total tenant debt envelope failed for {row.key}")
    hashes = {
        name: sorted(frame[name].dropna().astype(str).unique().tolist())
        for name in (
            "deployment_sha256",
            "simulator_sha256",
            "scheduler_core_sha256",
            "service_forecast_sha256",
        )
    }
    for name, values in hashes.items():
        if len(values) != 1:
            failures.append(f"mixed {name}: {values}")
    expected = {
        "deployment_sha256": protocol["deployment_sha256"],
        "simulator_sha256": protocol["simulator_sha256"],
        "scheduler_core_sha256": protocol["scheduler_core_sha256"],
        "service_forecast_sha256": protocol["service_forecast_sha256"],
    }
    for name, value in expected.items():
        if hashes.get(name) != [value]:
            failures.append(f"{name} does not match the protocol")
    registered_sources = protocol.get("execution_source_sha256")
    if not isinstance(registered_sources, dict) or not registered_sources:
        failures.append("protocol does not register execution source hashes")
    else:
        source_root = Path(__file__).parent
        for filename, registered_hash in sorted(registered_sources.items()):
            source_path = source_root / str(filename)
            if not source_path.is_file():
                failures.append(f"registered execution source is missing: {filename}")
                continue
            current_hash = sha256(source_path)
            if current_hash != registered_hash:
                failures.append(f"execution source hash changed after the campaign: {filename}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "campaign_directory": str(directory.resolve()),
        "run_count": int(len(frame)),
        "registered_run_count": int(protocol["task_count"]),
        "hashes": hashes,
        "execution_source_sha256": registered_sources,
        "failures": failures,
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, scenario, method), group in frame.groupby(
        ["family", "scenario", "method"], sort=True
    ):
        for metric in METRICS:
            n, mean, lower, upper = mean_ci(group[metric])
            rows.append(
                {
                    "family": family,
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, scenario), group in frame.groupby(["family", "scenario"], sort=True):
        reference = group[group.method == "scd"]
        if reference.empty:
            continue
        for method in sorted(set(group.method) - {"scd"}):
            challenger = group[group.method == method]
            for metric in METRICS:
                row = paired_row(
                    reference,
                    challenger,
                    family=family,
                    contrast="method_within_condition",
                    reference=f"{scenario}:scd",
                    challenger=f"{scenario}:{method}",
                    metric=metric,
                )
                if row is not None:
                    rows.append(row)

    primary = frame[(frame.family == "primary") & (frame.scenario == "primary")]
    cross_condition_families = {
        "ablation",
        "cache_capacity",
        "cache_identity",
        "cache_memory",
        "cache_ttl",
        "compatibility",
        "deadline_definition",
        "energy",
        "network",
        "payload",
        "power_uncertainty",
        "service_calibration",
        "thermal_model",
        "topology",
        "variability",
    }
    for (family, scenario, method), challenger in frame.groupby(
        ["family", "scenario", "method"], sort=True
    ):
        if family not in cross_condition_families:
            continue
        reference = primary[primary.method == ("scd" if family == "ablation" else method)]
        if reference.empty:
            continue
        for metric in METRICS:
            row = paired_row(
                reference,
                challenger,
                family=family,
                contrast="condition_vs_nominal",
                reference=f"primary:{reference.method.iloc[0]}",
                challenger=f"{scenario}:{method}",
                metric=metric,
            )
            if row is not None:
                rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm_within_family"] = np.nan
    for _, index in result.groupby("family").groups.items():
        indices = np.asarray(list(index), dtype=np.int64)
        result.loc[indices, "p_holm_within_family"] = holm(
            result.loc[indices, "p_unadjusted"].to_numpy(dtype=np.float64)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.campaign / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frame = load_runs(args.campaign)
    audit_report = audit(frame, protocol, args.campaign)
    if audit_report["status"] != "PASS":
        raise AssertionError(json.dumps(audit_report, indent=2))
    summary = summarize(frame)
    paired = contrasts(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.sort_values("key").to_csv(args.out / "verified_runs.csv", index=False)
    summary.to_csv(args.out / "descriptive_summary.csv", index=False)
    paired.to_csv(args.out / "paired_contrasts.csv", index=False)
    report = {
        "schema": "tnsm.strict-statistical-summary.v2",
        "analysis_source_sha256": sha256(Path(__file__)),
        "multiplicity_policy": "All registered stochastic contrasts remain. All-zero or undefined differences contribute p=1. Single-reconstruction controls are descriptive only.",
        **audit_report,
        "metrics": list(METRICS),
        "summary_rows": int(len(summary)),
        "paired_contrast_rows": int(len(paired)),
        "multiplicity": "Holm adjustment across all registered metric contrasts within each family",
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_sha256": sha256(Path(__file__)),
    }
    (args.out / "audit_and_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
