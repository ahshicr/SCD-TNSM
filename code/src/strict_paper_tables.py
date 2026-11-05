from __future__ import annotations

"""Build manuscript numbers and tables from source-locked campaign records."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.strict_paper_figures import load_campaign
from src.strict_config import load_deployment


METHOD_LABELS = {
    "scd": "SCD",
    "dpp_matched": "Matched DPP",
    "compute_first": "Compute-first",
    "energy_first": "Energy-first",
    "round_robin": "Round-robin",
    "scd_no_network": "No network penalty",
    "scd_independent_network": "Independent link state",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: Iterable[float]) -> tuple[int, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return 1, mean, math.nan
    half = float(stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
    return len(array), mean, half


def format_mean(values: Iterable[float], digits: int = 2) -> str:
    _, mean, half = mean_ci(values)
    if not np.isfinite(half):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {half:.{digits}f}"


def write_table(path: Path, header: str, rows: list[str]) -> None:
    if header.startswith(r"\toprule "):
        header = header[len(r"\toprule ") :]
    path.write_text("\n".join([header, r"\midrule", *rows, r"\bottomrule", ""]), encoding="utf-8")


def compact_range(values: Iterable[float], digits: int) -> str:
    array = np.asarray(list(values), dtype=np.float64)
    lower = float(array.min())
    upper = float(array.max())
    if np.isclose(lower, upper, rtol=0.0, atol=0.5 * 10 ** (-digits)):
        return f"{lower:.{digits}f}"
    return f"{lower:.{digits}f}--{upper:.{digits}f}"


def tex_probability(value: float | None) -> str:
    if value is None:
        return r"\text{none}"
    coefficient, exponent = f"{value:.0e}".split("e")
    exponent_value = int(exponent)
    if coefficient == "1":
        return rf"10^{{{exponent_value}}}"
    return rf"{coefficient}\times10^{{{exponent_value}}}"


def paired_effect(
    frame: pd.DataFrame,
    reference: str,
    challenger: str,
    metric: str,
    scale: float = 1.0,
) -> dict[str, float | int]:
    left = frame[frame.method == reference][["seed", metric]]
    right = frame[frame.method == challenger][["seed", metric]]
    pair = left.merge(right, on="seed", suffixes=("_reference", "_challenger"))
    difference = scale * (
        pair[f"{metric}_challenger"].to_numpy(dtype=float)
        - pair[f"{metric}_reference"].to_numpy(dtype=float)
    )
    n, mean, half = mean_ci(difference)
    return {"n": n, "mean": mean, "ci95_half": half}


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def macro_value(values: Iterable[float], digits: int, scale: float = 1.0) -> tuple[str, str]:
    _, mean, half = mean_ci(scale * np.asarray(list(values), dtype=np.float64))
    return f"{mean:.{digits}f}", f"{half:.{digits}f}"


def paired_values(
    frame: pd.DataFrame,
    left_method: str,
    right_method: str,
    metric: str,
    scale: float = 1.0,
) -> np.ndarray:
    left = frame[frame.method == left_method][["seed", metric]]
    right = frame[frame.method == right_method][["seed", metric]]
    pair = left.merge(right, on="seed", suffixes=("_left", "_right"))
    return scale * (
        pair[f"{metric}_left"].to_numpy(dtype=np.float64)
        - pair[f"{metric}_right"].to_numpy(dtype=np.float64)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--deadline", type=Path, required=True)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--breakdown", type=Path, required=True)
    parser.add_argument("--measured-startup", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_frame, main_protocol = load_campaign(args.campaign)
    network_frame, network_protocol = load_campaign(args.network)
    deadline_frame, deadline_protocol = load_campaign(args.deadline)
    load_frame, load_protocol = load_campaign(args.load)
    breakdown_frame, breakdown_protocol = load_campaign(args.breakdown)
    startup_frame, startup_protocol = load_campaign(args.measured_startup)
    deployment = load_deployment(args.deployment)
    if deployment.source_sha256 != main_protocol.get("deployment_sha256"):
        raise ValueError("table deployment does not match the main campaign")
    primary = main_frame[
        (main_frame.family == "primary") & (main_frame.scenario == "primary")
    ].copy()
    class_labels = {
        "pi5": "Pi 5",
        "pi4": "Pi 4",
        "pi3": "Pi 3",
        "pi2": "Pi 2",
        "rdk_x5": "RDK X5",
        "jetson_nano": "Jetson Nano",
    }
    included_ports = set(int(value) for value in deployment.port)
    node_records = [
        item
        for item in deployment.raw["nodes"]
        if int(item["port"]) in included_ports
    ]
    platform_rows: list[str] = []
    for class_name in ("pi5", "pi4", "pi3", "pi2", "rdk_x5", "jetson_nano"):
        members = [item for item in node_records if item["device_class"] == class_name]
        indices = [
            int(np.where(deployment.port == int(item["port"]))[0][0])
            for item in members
        ]
        egress_rates = [
            deployment.link_rate_bps[
                deployment.route_links[index, int(deployment.route_length[index]) - 1]
            ]
            / 1.0e6
            for index in indices
        ]
        evidence_text = " ".join(
            str(item[key]).lower()
            for item in members
            for key in ("power_evidence", "thermal_evidence", "startup_evidence")
        )
        if "bracket" in evidence_text:
            evidence = "sensitivity bracket"
        elif "transferred" in evidence_text and "direct" in evidence_text:
            evidence = "direct and transfer"
        elif "transferred" in evidence_text:
            evidence = "same-model transfer"
        else:
            evidence = "direct"
        platform_rows.append(
            f"{class_labels[class_name]} & {len(members)} & "
            f"{compact_range((deployment.service_rate[index] for index in indices), 3)} & "
            f"{compact_range(egress_rates, 1)} & "
            f"{compact_range((deployment.idle_w[index] for index in indices), 2)}/"
            f"{compact_range((deployment.incremental_w[index] for index in indices), 2)} & "
            f"{compact_range((math.exp(deployment.startup_mu_log_s[index] + 0.5 * deployment.startup_sigma_log_s[index] ** 2) for index in indices), 2)} & "
            f"{evidence} " + r"\\"
        )
    write_table(
        args.out / "platform_results.tex",
        r"\toprule Class & $n$ & Rate rel. & Egress Mb/s & $p^0/p^1$ W & Cold mean (s) & Evidence \\",
        platform_rows,
    )
    primary_order = ["scd", "dpp_matched", "compute_first", "energy_first", "round_robin"]
    nominal_rows: list[str] = []
    for method in primary_order:
        rows = primary[primary.method == method]
        nominal_rows.append(
            " & ".join(
                [
                    METHOD_LABELS[method],
                    format_mean(100.0 * rows.success_ratio, 2),
                    format_mean(100.0 * rows.miss_ratio, 2),
                    format_mean(rows.mean_latency_ms, 1),
                    format_mean(rows.p95_latency_ms, 1),
                    format_mean(rows.task_energy_mj, 3),
                    format_mean(rows.supplied_energy_mj, 3),
                    format_mean(100.0 * rows.actual_cold_start_ratio, 2),
                    format_mean(rows.jain_tenant_success, 4),
                ]
            )
            + r" \\"
        )
    write_table(
        args.out / "nominal_results.tex",
        r"\toprule Method & Offered success (\%) & Admitted miss (\%) & Mean (ms) & P95 (ms) & Task (MJ) & Supply (MJ) & Cold (\%) & Jain \\",
        nominal_rows,
    )

    sampling_specs = [
        ("Binomial, uniform", "primary", "primary"),
        ("Unbiased rounding", "sampling", "unbiased_rounding"),
        ("Binomial, stratified", "arrival_reconstruction", "stratified_within_minute"),
        ("Deterministic rounding", "sampling", "deterministic_rounding"),
    ]
    sampling_rows: list[str] = []
    for label, family, scenario in sampling_specs:
        rows = main_frame[
            (main_frame.family == family)
            & (main_frame.scenario == scenario)
            & (main_frame.method == "scd")
        ]
        sampling_rows.append(
            f"{label} & {int(round(rows.evaluated_requests.mean())):,} & "
            f"{format_mean(100.0 * rows.success_ratio, 2)} & "
            f"{format_mean(100.0 * rows.actual_cold_start_ratio, 2)} & "
            f"{format_mean(rows.mean_peak_aoi_ms / 1000.0, 1)} & "
            f"{format_mean(rows.jain_tenant_success, 4)} " + r"\\"
        )
    write_table(
        args.out / "sampling_results.tex",
        r"\toprule Reconstruction & Requests & Success (\%) & Cold (\%) & Mean peak AoI (s) & Jain \\",
        sampling_rows,
    )

    load_rows: list[str] = []
    load_sources = [
        (r"$0.5\times$", load_frame[load_frame.scenario == "load_0p5"]),
        (r"$1\times$", primary),
        (r"$2\times$", load_frame[load_frame.scenario == "load_2"]),
    ]
    for load_label, source in load_sources:
        for method in ("scd", "dpp_matched", "compute_first"):
            rows = source[source.method == method]
            load_rows.append(
                f"{load_label} & {METHOD_LABELS[method]} & "
                f"{int(round(rows.evaluated_requests.mean())):,} & "
                f"{format_mean(100.0 * rows.success_ratio, 2)} & "
                f"{format_mean(rows.mean_latency_ms, 1)} & "
                f"{format_mean(rows.task_energy_mj * 1.0e6 / rows.evaluated_requests, 3)} "
                + r"\\"
            )
    write_table(
        args.out / "load_scaling_results.tex",
        r"\toprule Load & Method & Offered requests & Success (\%) & Mean (ms) & Task (J/offered) \\",
        load_rows,
    )

    breakdown_rows: list[str] = []
    breakdown_probabilities = sorted(
        {
            float(Path(path).name.split("_p", 1)[1].split("_seed", 1)[0].replace("em", "e-"))
            for path in breakdown_frame.trace
        }
    )
    first_failure: float | None = None
    last_zero: float | None = None
    for probability in breakdown_probabilities:
        marker = f"breakdown_p{probability:.0e}".replace("+", "").replace("-", "m")
        rows = breakdown_frame[
            (breakdown_frame.scenario == marker) & (breakdown_frame.method == "scd")
        ]
        strict_misses = rows.admitted_requests.astype(int) - rows.on_time_requests.astype(int)
        any_failure = bool((strict_misses > 0).any())
        if any_failure and first_failure is None:
            first_failure = probability
        if not any_failure:
            last_zero = probability
        breakdown_rows.append(
            f"${tex_probability(probability)}$ & {int(round(rows.evaluated_requests.mean())):,} & "
            f"{int(strict_misses.min())}--{int(strict_misses.max())} & "
            f"{format_mean(100.0 * rows.success_ratio, 3)} & "
            f"{'yes' if any_failure else 'no'} " + r"\\"
        )
    write_table(
        args.out / "breakdown_results.tex",
        r"\toprule Native probability & Offered requests & Strict misses, range & Success (\%) & Any miss \\",
        breakdown_rows,
    )

    capacity_specs = [(f"{slot} slot" if slot == 1 else f"{slot} slots", f"cache_slots_{slot}") for slot in (1, 2, 4, 8, 16)]
    capacity_specs.append(("Memory limit", "primary"))
    capacity_rows: list[str] = []
    for label, scenario in capacity_specs:
        family = "primary" if scenario == "primary" else "cache_capacity"
        rows = main_frame[
            (main_frame.family == family)
            & (main_frame.scenario == scenario)
            & (main_frame.method == "scd")
        ]
        capacity_rows.append(
            f"{label} & {format_mean(100.0 * rows.success_ratio, 2)} & "
            f"{format_mean(100.0 * rows.actual_cold_start_ratio, 2)} & "
            f"{format_mean(rows.predicted_cache_evictions, 0)} & "
            f"{format_mean(rows.actual_cache_evictions, 0)} " + r"\\"
        )
    write_table(
        args.out / "cache_capacity_results.tex",
        r"\toprule Capacity rule & Success (\%) & Cold (\%) & Predicted evictions & Actual evictions \\",
        capacity_rows,
    )

    ablation_rows: list[str] = []
    for method, label in (
        ("scd", "SCD"),
        ("scd_no_forecast", "No forecast"),
        ("scd_static_weights", "Static weights"),
        ("scd_no_thermal_term", "No thermal term"),
    ):
        if method == "scd":
            rows = primary[primary.method == method]
        else:
            rows = main_frame[(main_frame.family == "ablation") & (main_frame.method == method)]
        ablation_rows.append(
            f"{label} & {format_mean(100.0 * rows.success_ratio, 3)} & "
            f"{format_mean(rows.mean_latency_ms, 1)} & "
            f"{format_mean(rows.task_energy_mj, 4)} & "
            f"{format_mean(rows.mean_peak_aoi_ms / 1000.0, 2)} & "
            f"{format_mean(rows.peak_temperature_c, 2)} " + r"\\"
        )
    write_table(
        args.out / "component_ablation_results.tex",
        r"\toprule Method & Success (\%) & Mean (ms) & Task energy (MJ) & Peak AoI (s) & Peak temp. ($^\circ$C) \\",
        ablation_rows,
    )

    energy_rows: list[str] = []
    for factor, label in ((0.30, "0.30"), (0.10, "0.10")):
        scenario = f"energy_{factor:g}"
        for method in ("scd", "scd_no_debt", "scd_shared_energy", "dpp_matched", "compute_first"):
            rows = main_frame[
                (main_frame.family == "energy")
                & (main_frame.scenario == scenario)
                & (main_frame.method == method)
            ]
            method_label = {
                **METHOD_LABELS,
                "scd_no_debt": "No debt",
                "scd_shared_energy": "Shared energy",
            }[method]
            rejected = 100.0 * rows.rejected_requests / rows.evaluated_requests
            energy_rows.append(
                f"{label} & {method_label} & {format_mean(100.0 * rows.success_ratio, 3)} & "
                f"{format_mean(rejected, 3)} & {format_mean(rows.task_energy_mj, 4)} & "
                f"{format_mean(rows.maximum_per_device_energy_debt_j, 2)} " + r"\\"
            )
    write_table(
        args.out / "energy_pressure_results.tex",
        r"\toprule Budget factor & Method & Success (\%) & Rejected (\%) & Task energy (MJ) & Max. debt (J) \\",
        energy_rows,
    )

    tenant_rows: list[str] = []
    for factor, label in ((0.30, "0.30"), (0.10, "0.10")):
        scenario = f"energy_{factor:g}"
        for method, method_label in (("scd", "Isolated SCD"), ("scd_shared_energy", "Shared energy")):
            rows = main_frame[
                (main_frame.family == "energy")
                & (main_frame.scenario == scenario)
                & (main_frame.method == method)
            ]
            tenant_rows.append(
                f"{label} & {method_label} & {format_mean(100.0 * rows.success_ratio, 3)} & "
                f"{format_mean(rows.jain_tenant_success, 4)} & "
                f"{format_mean(100.0 * rows.minimum_tenant_success_ratio, 3)} & "
                f"{format_mean(rows.maximum_per_device_tenant_energy_debt_j, 2)} " + r"\\"
            )
    write_table(
        args.out / "tenant_isolation_results.tex",
        r"\toprule Budget factor & Policy & Success (\%) & Jain & Minimum tenant (\%) & Max. tenant debt (J) \\",
        tenant_rows,
    )

    flood_rows: list[str] = []
    for multiplier in (64, 256):
        scenario_guarded = f"attack_{multiplier}_guarded"
        scenario_unguarded = f"attack_{multiplier}_unguarded"
        for method, scenario, label in (
            ("scd", scenario_guarded, "SCD guarded"),
            ("compute_first", scenario_guarded, "Compute-first guarded"),
            ("scd", scenario_unguarded, "SCD unguarded"),
        ):
            rows = main_frame[
                (main_frame.family == "request_flood")
                & (main_frame.scenario == scenario)
                & (main_frame.method == method)
            ]
            flood_rows.append(
                f"{multiplier}$\\times$ & {label} & "
                f"{format_mean(100.0 * rows.legitimate_success_ratio, 3)} & "
                f"{format_mean(100.0 * rows.attacker_admission_ratio, 3)} & "
                f"{format_mean(rows.task_energy_mj, 4)} & "
                f"{format_mean(rows.maximum_per_device_energy_debt_j, 2)} " + r"\\"
            )
    write_table(
        args.out / "request_flood_results.tex",
        r"\toprule Flood & Policy & Legitimate success (\%) & Attack admitted (\%) & Task energy (MJ) & Max. debt (J) \\",
        flood_rows,
    )

    uncertainty_rows: list[str] = []
    uncertainty_specs = (
        ("primary", "primary", "scd", "CV 0.15, margin"),
        ("power_uncertainty", "power_cv_0", "scd", "CV 0, margin"),
        ("power_uncertainty", "power_cv_0.4", "scd", "CV 0.40, margin"),
        ("power_uncertainty", "power_margin_removed", "scd", "CV 0.15, no margin"),
    )
    for family, scenario, method, label in uncertainty_specs:
        rows = main_frame[
            (main_frame.family == family)
            & (main_frame.scenario == scenario)
            & (main_frame.method == method)
        ]
        uncertainty_rows.append(
            f"{label} & {format_mean(100.0 * rows.success_ratio, 3)} & "
            f"{format_mean(100.0 * rows.energy_underestimation_ratio, 3)} & "
            f"{format_mean(rows.energy_shortfall_j / 1000.0, 2)} & "
            f"{format_mean(rows.task_energy_mj, 4)} & "
            f"{format_mean(rows.maximum_per_device_energy_debt_j, 2)} " + r"\\"
        )
    write_table(
        args.out / "power_uncertainty_results.tex",
        r"\toprule Condition & Success (\%) & Underestimated (\%) & Shortfall (kJ) & Task energy (MJ) & Max. debt (J) \\",
        uncertainty_rows,
    )

    condition_labels = {
        "stress_100mbps_bg12_payload4": r"100 Mb/s, $4\times$ payload",
        "stress_10mbps_bg12_payload4": r"10 Mb/s, $4\times$ payload",
        "stress_10mbps_bg12_payload16": r"10 Mb/s, $16\times$ payload",
    }
    network_rows: list[str] = []
    for scenario, label in condition_labels.items():
        rows = network_frame[network_frame.scenario == scenario]
        for method in ("scd", "scd_no_network", "scd_independent_network"):
            group = rows[rows.method == method]
            network_rows.append(
                f"{label} & {METHOD_LABELS[method]} & "
                f"{format_mean(100.0 * group.success_ratio, 3)} & "
                f"{format_mean(group.mean_latency_ms / 1000.0, 2)} & "
                f"{format_mean(group.p95_network_queue_ms, 2)} " + r"\\"
            )
    write_table(
        args.out / "network_stress_results.tex",
        r"\toprule Condition & Policy & Success (\%) & Mean latency (s) & P95 queue (ms) \\",
        network_rows,
    )

    deadline_rows: list[str] = []
    for scenario, label in condition_labels.items():
        base = network_frame[(network_frame.scenario == scenario) & (network_frame.method == "scd")]
        variants = [
            ("Best effort", base),
            (r"$4\times$ ceiling", deadline_frame[(deadline_frame.scenario == f"{scenario}_grace4") & (deadline_frame.method == "scd")]),
            ("Strict reject", deadline_frame[(deadline_frame.scenario == f"{scenario}_strict") & (deadline_frame.method == "scd")]),
        ]
        for mode, group in variants:
            deadline_rows.append(
                f"{label} & {mode} & {format_mean(100.0 * group.success_ratio, 3)} & "
                f"{format_mean(100.0 * group.deadline_ceiling_success_ratio, 3)} & "
                f"{format_mean(group.p95_network_queue_ms, 2)} & "
                f"{format_mean(100.0 * group.rejected_requests / group.evaluated_requests, 3)} " + r"\\"
            )
    write_table(
        args.out / "deadline_guard_results.tex",
        r"\toprule Condition & Mode & Strict success (\%) & Ceiling success (\%) & P95 queue (ms) & Rejected (\%) \\",
        deadline_rows,
    )

    startup_rows: list[str] = []
    for ttl in (30, 60, 300, 900):
        scenario = f"measured_central_ttl_{ttl}"
        for method in ("scd", "dpp_matched", "compute_first"):
            rows = startup_frame[
                (startup_frame.scenario == scenario) & (startup_frame.method == method)
            ]
            if len(rows) != 5:
                raise ValueError(f"measured-startup cell {scenario}/{method} has {len(rows)} runs")
            startup_rows.append(
                f"{ttl} & {METHOD_LABELS[method]} & "
                f"{format_mean(100.0 * rows.success_ratio, 2)} & "
                f"{format_mean(rows.mean_latency_ms, 1)} & "
                f"{format_mean(rows.p95_latency_ms, 1)} & "
                f"{format_mean(100.0 * rows.actual_cold_start_ratio, 2)} " + r"\\"
            )
    write_table(
        args.out / "measured_startup_table.tex",
        r"\toprule TTL (s) & Method & Success (\%) & Mean (ms) & P95 (ms) & Cold (\%) \\",
        startup_rows,
    )

    exact_path = args.exact / "exact_benchmark_runs.csv"
    exact = pd.read_csv(exact_path)
    exact_rows = [
        f"{int(row.seed)} & {int(row.enumerated_assignments):,} & {row.exact_objective:.6f} & "
        f"{row.greedy_objective:.6f} & {row.greedy_gap_percent:.2f} " + r"\\"
        for row in exact.itertuples(index=False)
    ]
    write_table(
        args.out / "exact_benchmark_results.tex",
        r"\toprule Seed & Assignments & Exact objective & Greedy objective & Gap (\%) \\",
        exact_rows,
    )

    primary_export = primary.copy()
    primary_export["task_energy_j_per_offered"] = (
        primary_export.task_energy_mj * 1.0e6 / primary_export.evaluated_requests
    )
    primary_export.to_csv(args.out / "nominal_results_source.csv", index=False)
    pd.concat(
        [
            main_frame,
            network_frame,
            deadline_frame,
            load_frame,
            breakdown_frame,
            startup_frame,
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        args.out / "all_registered_runs.csv", index=False
    )

    numbers: dict[str, object] = {
        "schema": "tnsm.paper-values.v1",
        "primary_run_count": int(len(main_frame)),
        "network_run_count": int(len(network_frame)),
        "deadline_run_count": int(len(deadline_frame)),
        "load_run_count": int(len(load_frame)),
        "breakdown_run_count": int(len(breakdown_frame)),
        "measured_startup_run_count": int(len(startup_frame)),
        "total_registered_runs": int(
            len(main_frame)
            + len(network_frame)
            + len(deadline_frame)
            + len(load_frame)
            + len(breakdown_frame)
            + len(startup_frame)
        ),
        "primary_requests_mean": int(round(primary[primary.method == "scd"].evaluated_requests.mean())),
        "primary_scd": {},
        "primary_effects_challenger_minus_scd": {},
        "exact": json.loads((args.exact / "exact_benchmark_summary.json").read_text(encoding="utf-8")),
    }
    scd = primary[primary.method == "scd"]
    for metric in (
        "success_ratio",
        "miss_ratio",
        "mean_latency_ms",
        "p95_latency_ms",
        "task_energy_mj",
        "supplied_energy_mj",
        "mean_peak_aoi_ms",
        "p95_peak_aoi_ms",
        "jain_tenant_success",
        "actual_cold_start_ratio",
        "peak_temperature_c",
    ):
        n, mean, half = mean_ci(scd[metric])
        numbers["primary_scd"][metric] = {"n": n, "mean": mean, "ci95_half": half}
    for challenger in primary_order[1:]:
        effects: dict[str, object] = {}
        for metric, scale in (
            ("success_ratio", 100.0),
            ("miss_ratio", 100.0),
            ("mean_latency_ms", 1.0),
            ("task_energy_mj", 1.0),
            ("supplied_energy_mj", 1.0),
            ("jain_tenant_success", 100.0),
            ("actual_cold_start_ratio", 100.0),
        ):
            effects[metric] = paired_effect(primary, "scd", challenger, metric, scale)
        numbers["primary_effects_challenger_minus_scd"][challenger] = effects

    protocol_records = {
        "main": (args.campaign, main_protocol),
        "network": (args.network, network_protocol),
        "deadline": (args.deadline, deadline_protocol),
        "load": (args.load, load_protocol),
        "breakdown": (args.breakdown, breakdown_protocol),
        "measured_startup": (args.measured_startup, startup_protocol),
    }
    numbers["protocols"] = {
        name: {
            "directory": str(path.resolve()),
            "protocol_sha256": protocol["protocol_sha256"],
            "task_count": int(protocol["task_count"]),
            "execution_source_sha256": protocol.get("execution_source_sha256", {}),
        }
        for name, (path, protocol) in protocol_records.items()
    }
    numbers_path = args.out / "paper_values.json"
    numbers_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    macros: dict[str, str] = {
        "StrictMainRuns": str(len(main_frame)),
        "StrictNetworkRuns": str(len(network_frame)),
        "StrictDeadlineRuns": str(len(deadline_frame)),
        "StrictLoadRuns": str(len(load_frame)),
        "StrictBreakdownRuns": str(len(breakdown_frame)),
        "StrictMeasuredStartupRuns": str(len(startup_frame)),
        "StrictPrimaryRequests": f"{int(round(primary[primary.method == 'scd'].evaluated_requests.mean())):,}",
        "StrictTotalRuns": str(
            len(main_frame)
            + len(network_frame)
            + len(deadline_frame)
            + len(load_frame)
            + len(breakdown_frame)
            + len(startup_frame)
        ),
        "ExactAssignmentsPerSeed": f"{int(numbers['exact']['assignments_per_seed']):,}",
        "ExactAssignmentsTotal": f"{int(numbers['exact']['assignments_per_seed']) * len(numbers['exact']['seeds']):,}",
        "ExactGreedyGapPct": f"{float(numbers['exact']['mean_greedy_gap_percent']):.2f}",
        "BreakdownFirstFailureProbability": tex_probability(first_failure),
        "BreakdownLastZeroProbability": tex_probability(last_zero),
    }
    for method, prefix in (
        ("scd", "SCD"),
        ("dpp_matched", "DPP"),
        ("compute_first", "Compute"),
        ("energy_first", "Energy"),
        ("round_robin", "RR"),
    ):
        rows = primary[primary.method == method]
        for suffix, metric, digits, scale in (
            ("SuccessPct", "success_ratio", 2, 100.0),
            ("MissPct", "miss_ratio", 2, 100.0),
            ("MeanLatencyMs", "mean_latency_ms", 1, 1.0),
            ("TaskEnergyMJ", "task_energy_mj", 3, 1.0),
            ("SupplyEnergyMJ", "supplied_energy_mj", 3, 1.0),
            ("PeakAoIS", "mean_peak_aoi_ms", 1, 0.001),
            ("Jain", "jain_tenant_success", 4, 1.0),
            ("ColdPct", "actual_cold_start_ratio", 2, 100.0),
        ):
            mean, half = macro_value(rows[metric], digits, scale)
            macros[f"{prefix}{suffix}"] = mean
            macros[f"{prefix}{suffix}CI"] = half

    for challenger, prefix in (
        ("compute_first", "SCDMinusCompute"),
        ("dpp_matched", "SCDMinusDPP"),
        ("energy_first", "SCDMinusEnergy"),
    ):
        for suffix, metric, digits, scale in (
            ("SuccessPP", "success_ratio", 3, 100.0),
            ("LatencyMs", "mean_latency_ms", 2, 1.0),
            ("TaskEnergyMJ", "task_energy_mj", 4, 1.0),
            ("SupplyEnergyMJ", "supplied_energy_mj", 4, 1.0),
            ("JainPoints", "jain_tenant_success", 3, 100.0),
            ("PeakAoIS", "mean_peak_aoi_ms", 2, 0.001),
        ):
            mean, half = macro_value(
                paired_values(primary, "scd", challenger, metric, scale=scale),
                digits,
            )
            macros[f"{prefix}{suffix}"] = mean
            macros[f"{prefix}{suffix}CI"] = half

    for name, challenger, metric, digits in (
        ("SCDComputeLatencyReductionMs", "compute_first", "mean_latency_ms", 2),
        ("SCDDPPTaskEnergyReductionMJ", "dpp_matched", "task_energy_mj", 4),
        ("SCDDPPSupplyEnergyReductionMJ", "dpp_matched", "supplied_energy_mj", 4),
    ):
        mean, half = macro_value(
            paired_values(primary, challenger, "scd", metric, scale=1.0), digits
        )
        macros[name] = mean
        macros[f"{name}CI"] = half

    compute_pairs = primary[primary.method == "scd"][
        ["seed", "mean_latency_ms", "task_energy_mj", "supplied_energy_mj"]
    ].merge(
        primary[primary.method == "compute_first"][
            ["seed", "mean_latency_ms", "task_energy_mj", "supplied_energy_mj"]
        ],
        on="seed",
        suffixes=("_scd", "_compute"),
    )
    for suffix, metric, digits in (
        ("LatencyPct", "mean_latency_ms", 2),
        ("TaskEnergyPct", "task_energy_mj", 2),
        ("SupplyEnergyPct", "supplied_energy_mj", 2),
    ):
        relative = 100.0 * (
            compute_pairs[f"{metric}_scd"] - compute_pairs[f"{metric}_compute"]
        ) / compute_pairs[f"{metric}_compute"]
        mean, half = macro_value(relative, digits)
        macros[f"SCDMinusCompute{suffix}"] = mean
        macros[f"SCDMinusCompute{suffix}CI"] = half

    for scenario, prefix in (("load_0p5", "HalfLoad"), ("load_2", "DoubleLoad")):
        subset = load_frame[(load_frame.scenario == scenario) & (load_frame.method == "scd")]
        for suffix, metric, digits, scale in (
            ("SuccessPct", "success_ratio", 2, 100.0),
            ("MissPct", "miss_ratio", 2, 100.0),
            ("LatencyMs", "mean_latency_ms", 1, 1.0),
            ("TaskJPerOffered", "task_energy_mj", 3, 1.0e6),
        ):
            values = subset[metric]
            if suffix == "TaskJPerOffered":
                values = subset.task_energy_mj / subset.evaluated_requests
            mean, half = macro_value(values, digits, scale)
            macros[f"{prefix}{suffix}"] = mean
            macros[f"{prefix}{suffix}CI"] = half

    network_prefixes = {
        "stress_100mbps_bg12_payload4": "StressHundredPayloadFour",
        "stress_10mbps_bg12_payload4": "StressTenPayloadFour",
        "stress_10mbps_bg12_payload16": "StressTenPayloadSixteen",
    }
    for scenario, prefix in network_prefixes.items():
        subset = network_frame[network_frame.scenario == scenario]
        for challenger, challenger_prefix in (
            ("scd_no_network", "NoNetworkMinusSCD"),
            ("scd_independent_network", "IndependentMinusSCD"),
        ):
            for suffix, metric, digits, scale in (
                ("SuccessPP", "success_ratio", 3, 100.0),
                ("LatencyMs", "mean_latency_ms", 2, 1.0),
                ("QueueMs", "p95_network_queue_ms", 2, 1.0),
            ):
                mean, half = macro_value(
                    paired_values(subset, challenger, "scd", metric, scale), digits
                )
                macros[f"{prefix}{challenger_prefix}{suffix}"] = mean
                macros[f"{prefix}{challenger_prefix}{suffix}CI"] = half

    invalid_macro_names = sorted(name for name in macros if not name.isalpha())
    if invalid_macro_names:
        raise ValueError(f"LaTeX control words must contain letters only: {invalid_macro_names}")
    macro_path = args.out / "paper_values_macros.tex"
    macro_path.write_text(
        "\n".join(
            rf"\newcommand{{\{name}}}{{{value}}}"
            for name, value in sorted(macros.items())
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = {
        "schema": "tnsm.paper-tables.v1",
        "status": "PASS",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__)),
        "paper_values_sha256": sha256(numbers_path),
        "exact_runs_sha256": sha256(exact_path),
        "outputs": {
            path.name: sha256(path)
            for path in sorted(args.out.iterdir())
            if path.is_file() and path.name != "table_provenance.json"
        },
    }
    (args.out / "table_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "outputs": len(provenance["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
