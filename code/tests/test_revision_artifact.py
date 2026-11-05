from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "results" / "strict"
SEEDS = {str(seed) for seed in range(20260914, 20260919)}
CAMPAIGNS = {
    "campaign_formula_v7_20260914": 618,
    "network_stress_formula_v4_20260914": 75,
    "load_scaling_formula_v2_20260914": 30,
    "deadline_guard_network_stress_v4_20260914": 60,
    "breakdown_formula_v1_20260914": 165,
    "measured_startup_strict_v1_20260914": 180,
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_source_workload_preserves_raw_function_resolution() -> None:
    data = np.load(STRICT / "processed" / "workload_strict_v2.npz")
    assert data["counts"].shape == (9_000, 20_160)
    assert int(data["counts"].sum()) == 12_444_637_039
    assert data["duration_quantiles_ms"].shape == (9_000, 14, 7)
    assert data["app_ids"].shape == (9_000,)
    assert len(np.unique(data["function_keys"])) == 9_000
    assert np.isfinite(data["duration_quantiles_ms"]).all()
    assert (data["counts"] >= 0).all()


def test_forecast_artifact_and_held_out_metrics() -> None:
    base = ROOT / "results" / "revision" / "forecast_prequential_20260905"
    pred = np.load(base / "forecast_prequential.npz")
    assert pred["prediction"].shape == (20_160, 16)
    assert pred["target"].shape == (20_160, 16)
    assert np.isfinite(pred["prediction"]).all()
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"].startswith("daily expanding-window prequential")
    assert summary["minutes"] == 20_160
    assert summary["classes"] == 16
    metrics = summary["metrics_on_unseen_deployment_blocks"]
    assert metrics["model"]["rmse"] < metrics["persistence"]["rmse"]
    assert metrics["model"]["rmse"] < metrics["ewma"]["rmse"]


def test_primary_trace_registry_is_source_locked() -> None:
    registry = json.loads(
        (STRICT / "traces_v3" / "primary_trace_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["status"] == "PASS"
    assert registry["probability"] == 4.0e-4
    assert registry["trace_count"] == 16
    assert registry["seeds"] == list(range(20260914, 20260919))
    assert len({item["directory"] for item in registry["traces"]}) == 16
    assert all(len(item["manifest_sha256"]) == 64 for item in registry["traces"])


def test_all_formal_campaigns_are_complete_and_unique() -> None:
    combined: list[dict[str, str]] = []
    for directory, expected_count in CAMPAIGNS.items():
        campaign = STRICT / directory
        current = rows(campaign / "campaign_results.csv")
        assert len(current) == expected_count
        assert len({item["key"] for item in current}) == expected_count
        assert {item["seed"] for item in current} == SEEDS
        assert all(len(item["simulator_sha256"]) == 64 for item in current)
        assert all(len(item["scheduler_core_sha256"]) == 64 for item in current)
        assert {item["cache_identity_mode"] for item in current} <= {
            "function",
            "stratum",
        }
        assert all(
            item["cache_identity_mode"] == "function"
            or item["family"] == "cache_identity"
            for item in current
        )
        success = np.asarray([float(item["success_ratio"]) for item in current])
        assert ((0.0 <= success) & (success <= 1.0)).all()
        audit = json.loads(
            (campaign / "analysis" / "audit_and_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert audit["status"] == "PASS"
        combined.extend(current)
    assert len(combined) == 1128
    assert len({item["key"] for item in combined}) == 1128

    startup_root = STRICT / "measured_startup_strict_v1_20260914"
    startup = rows(startup_root / "campaign_results.csv")
    assert {
        profile: sum(item["startup_profile"] == profile for item in startup)
        for profile in ("measured_low", "measured_central", "measured_high")
    } == {"measured_low": 60, "measured_central": 60, "measured_high": 60}
    protocol = json.loads((startup_root / "protocol.json").read_text(encoding="utf-8"))
    evidence = dict(protocol["startup_profile_evidence"])
    registered_hash = evidence.pop("profile_sha256")
    calculated_hash = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert registered_hash == protocol["startup_profile_evidence_sha256"] == calculated_hash
    profile_hashes = {
        task["startup_profile"]: task["startup_profile_sha256"]
        for task in protocol["tasks"]
    }
    assert all(
        item["startup_profile_sha256"] == profile_hashes[item["startup_profile"]]
        for item in startup
    )


def test_paper_table_registry_matches_formal_campaigns() -> None:
    table_root = STRICT / "paper_tables_formula_v3_20260914"
    registered = rows(table_root / "all_registered_runs.csv")
    assert len(registered) == 1128
    assert len({item["key"] for item in registered}) == 1128
    provenance = json.loads(
        (table_root / "table_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "PASS"
    for name, expected_hash in provenance["outputs"].items():
        assert hashlib.sha256((table_root / name).read_bytes()).hexdigest() == expected_hash
    for name in (
        "nominal_results.tex",
        "sampling_results.tex",
        "load_scaling_results.tex",
        "breakdown_results.tex",
        "cache_capacity_results.tex",
        "component_ablation_results.tex",
        "energy_pressure_results.tex",
        "tenant_isolation_results.tex",
        "request_flood_results.tex",
        "power_uncertainty_results.tex",
        "network_stress_results.tex",
        "deadline_guard_results.tex",
        "exact_benchmark_results.tex",
        "measured_startup_table.tex",
        "paper_values_macros.tex",
    ):
        text = (table_root / name).read_text(encoding="utf-8")
        if name != "paper_values_macros.tex":
            lines = text.splitlines()
            assert lines and "&" in lines[0] and lines[0].endswith(r"\\")
            assert lines[1] == r"\midrule"
            assert "\t" not in text
        assert "\\\\midrule" not in text
        assert "\\\\bottomrule" not in text

    macros = (table_root / "paper_values_macros.tex").read_text(encoding="utf-8")
    all_macro_values = {
        line.split("{", 2)[1].rstrip("}"): float(line.split("{", 2)[2].rstrip("}"))
        for line in macros.splitlines()
        if line.startswith(r"\newcommand{\SCD")
    }
    assert np.isclose(
        all_macro_values[r"\SCDComputeLatencyReductionMs"],
        -all_macro_values[r"\SCDMinusComputeLatencyMs"],
    )
    assert np.isclose(
        all_macro_values[r"\SCDDPPTaskEnergyReductionMJ"],
        -all_macro_values[r"\SCDMinusDPPTaskEnergyMJ"],
    )
    assert np.isclose(
        all_macro_values[r"\SCDDPPSupplyEnergyReductionMJ"],
        -all_macro_values[r"\SCDMinusDPPSupplyEnergyMJ"],
    )
    primary_request_line = next(
        line for line in macros.splitlines() if line.startswith(r"\newcommand{\StrictPrimaryRequests}")
    )
    primary_requests = primary_request_line.split("{", 2)[2].rsplit("}", 1)[0]
    primary_row = (table_root / "sampling_results.tex").read_text(encoding="utf-8").splitlines()[2]
    assert primary_row.split("&", 2)[1].strip() == primary_requests

    macro_values = {
        line.split("{", 2)[1].rstrip("}"): float(line.split("{", 2)[2].rstrip("}"))
        for line in macros.splitlines()
        if line.startswith(r"\newcommand{\Stress")
        and not line.startswith(r"\newcommand{\StressTenPayloadFour")
        and not line.endswith("CI}")
    }
    for scenario in ("StressHundredPayloadFour", "StressTenPayloadSixteen"):
        assert macro_values[f"\\{scenario}IndependentMinusSCDSuccessPP"] < 0.0
        assert macro_values[f"\\{scenario}IndependentMinusSCDLatencyMs"] > 0.0
        assert macro_values[f"\\{scenario}IndependentMinusSCDQueueMs"] > 0.0

    startup_count_line = next(line for line in macros.splitlines()
                              if line.startswith(r"\newcommand{\StrictMeasuredStartupRuns}"))
    assert int(startup_count_line.split("{", 2)[2].rstrip("}")) == CAMPAIGNS[
        "measured_startup_strict_v1_20260914"
    ]


def test_performance_claims_follow_primary_results() -> None:
    primary_rows = [
        item
        for item in rows(
            STRICT / "campaign_formula_v7_20260914" / "campaign_results.csv"
        )
        if item["family"] == "primary" and item["scenario"] == "primary"
    ]
    methods = {item["method"] for item in primary_rows}
    means = {
        method: {
            metric: float(
                np.mean(
                    [
                        float(item[metric])
                        for item in primary_rows
                        if item["method"] == method
                    ]
                )
            )
            for metric in ("success_ratio", "mean_latency_ms", "jain_tenant_success", "task_energy_mj")
        }
        for method in methods
    }
    assert max(means, key=lambda method: means[method]["success_ratio"]) == "scd"
    assert min(means, key=lambda method: means[method]["mean_latency_ms"]) == "scd"
    assert max(means, key=lambda method: means[method]["jain_tenant_success"]) == "scd"

    # Validate the published quantities against complete run records without
    # requiring unpublished prose to be part of the public code artifact.
    macros=(STRICT/"paper_tables_formula_v3_20260914/paper_values_macros.tex").read_text(encoding="utf-8")
    def published(name):
        prefix=r"\newcommand{"+chr(92)+name+"}"
        line=next(line for line in macros.splitlines() if line.startswith(prefix))
        return float(line.split("{",2)[2].rstrip("}"))
    assert np.isclose(published("SCDComputeLatencyReductionMs"),
                      means["compute_first"]["mean_latency_ms"]-means["scd"]["mean_latency_ms"],atol=.005,rtol=0)
    assert np.isclose(published("SCDDPPTaskEnergyReductionMJ"),
                      means["dpp_matched"]["task_energy_mj"]-means["scd"]["task_energy_mj"],atol=.00005,rtol=0)
    assert np.isclose(published("SCDMinusDPPSuccessPP"),
                      100*(means["scd"]["success_ratio"]-means["dpp_matched"]["success_ratio"]),atol=.0005,rtol=0)


def test_figure_outputs_are_editable_and_source_backed() -> None:
    output = STRICT / "paper_figures_formula_v5_20260915"
    provenance = json.loads(
        (output / "figure_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "PASS"
    for name, expected_hash in provenance["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    for stem in (
        "fig1_architecture",
        "fig2_primary",
        "fig3_sampling_cache",
        "fig4_effects_stress",
        "figS_load_scaling",
        "figS_breakdown",
    ):
        assert (output / f"{stem}.pdf").stat().st_size > 10_000
        assert (output / f"{stem}.png").stat().st_size > 100_000
        svg = (output / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<svg" in svg and "<path" in svg and "<text" in svg
        if stem != "fig1_architecture":
            assert (output / f"{stem}_source.csv").stat().st_size > 1_000
        if stem in {"fig2_primary", "fig3_sampling_cache", "fig4_effects_stress"}:
            pixels = np.asarray(Image.open(output / f"{stem}.png").convert("RGB"))
            foreground_fraction = np.any(pixels < 245, axis=2).mean()
            assert foreground_fraction > 0.08


def test_exact_and_conformance_reports_pass() -> None:
    exact = json.loads(
        (
            STRICT
            / "exact_benchmark_suite_v2_20260914"
            / "exact_benchmark_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert exact["status"] == "PASS"
    assert exact["assignments_per_seed"] == 16_384
    assert exact["assignments_per_seed"] * len(exact["seeds"]) == 81_920
    assert exact["exact_dominates_greedy_all_seeds"] is True

    conformance = json.loads(
        (
            STRICT / "implementation_conformance_final_v16_20260914.json"
        ).read_text(encoding="utf-8")
    )
    assert conformance["status"] == "PASS"
    assert conformance["checks"] == 49
    assert conformance["passed"] == 49
    assert conformance["failed"] == 0
