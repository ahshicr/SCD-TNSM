from __future__ import annotations

"""Trace every strict-deployment number to a measurement or named model input.

The audit deliberately distinguishes direct measurements, same-class transfers,
and unmeasured model assumptions.  A transferred/modelled value is not promoted
to a direct measurement merely because its numeric value is internally valid.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT.parent / "manuscript"
PHYSICAL = ROOT / "results" / "revision" / "physical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment",
        type=Path,
        default=ROOT / "config" / "strict_deployment_20260914.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "strict" / "configuration_audit_20260914",
    )
    args = parser.parse_args()

    deployment = load_json(args.deployment)
    nodes = {int(row["port"]): row for row in deployment["nodes"]}
    links = {
        int(row["name"].rsplit("-", 1)[1]): row
        for row in deployment["links"]
        if str(row["name"]).startswith("h3c-port-")
    }
    rows: list[dict[str, Any]] = []

    def record(
        category: str,
        port: int,
        parameter: str,
        configured: Any,
        expected: Any,
        tolerance: float | None,
        evidence_class: str,
        source: Path | str,
        note: str = "",
    ) -> None:
        if tolerance is None:
            passed = configured == expected
            error = None
        else:
            error = abs(float(configured) - float(expected))
            passed = error <= tolerance
        rows.append(
            {
                "category": category,
                "port": port,
                "parameter": parameter,
                "configured": configured,
                "expected": expected,
                "absolute_error": error,
                "tolerance": tolerance,
                "evidence_class": evidence_class,
                "source": str(source),
                "note": note,
                "status": "PASS" if passed else "FAIL",
            }
        )

    capability_path = PHYSICAL / "capabilities_20260905.json"
    capabilities = {int(row["port"]): row for row in load_json(capability_path)["nodes"]}
    for port, node in nodes.items():
        measured = capabilities[port]["capabilities"]
        record("inventory", port, "model", node["model"], measured["model"], None,
               "direct", capability_path)
        record("inventory", port, "arch", node["arch"], measured["arch"], None,
               "direct", capability_path)
        match = re.search(r"MemTotal:\s+(\d+) kB", measured["memory"][0])
        if not match:
            raise ValueError(f"cannot parse MemTotal on port {port}")
        measured_mb = int(match.group(1)) / 1024.0
        record("inventory", port, "memory_total_mb", node["memory_total_mb"],
               measured_mb, 0.11, "direct", capability_path, "Linux MemTotal / 1024")

    service_path = PHYSICAL / "inference_calibration_v2_20260905" / "service_models.json"
    service_p4_path = (
        PHYSICAL / "inference_calibration_port04_repeat_20260909" / "service_models.json"
    )
    service_rows = load_json(service_path)["batch_statistics"]
    service_rows += load_json(service_p4_path)["batch_statistics"]
    service_mean = {
        (int(row["port"]), int(row["batch"])): float(row["service_s"]["mean"])
        for row in service_rows
    }
    for batch, profile in ((1, "batch1"), (16, "batch16"), (64, "batch64")):
        reference = service_mean[(23, batch)]
        for port, node in nodes.items():
            expected = reference / service_mean[(port, batch)]
            configured = (
                node["service_rate"]
                if batch == 16
                else deployment["service_rate_profiles"][profile][str(port)]
            )
            source = service_p4_path if port == 4 else service_path
            record("service", port, f"relative_rate_{profile}", configured, expected,
                   5.1e-5, "direct", source,
                   "Port-23 session mean divided by board session mean")

    transport_path = (
        PHYSICAL / "activation_transport_isolated_20260905" /
        "activation_transport_analysis.json"
    )
    transport = {
        int(row["port"]): row["controller_transport_fit"]
        for row in load_json(transport_path)["transport"]
    }
    for port, link in links.items():
        source_port = 1 if port == 4 else port
        evidence = "same-model transfer" if port == 4 else "direct"
        fit = transport[source_port]
        record("transport", port, "effective_mbps", link["rate_mbps"],
               fit["effective_mbps"], 5.1e-3, evidence, transport_path,
               fit["scope"])
        record("transport", port, "fixed_overhead_ms", link.get("fixed_overhead_ms", 0.0),
               fit["fixed_overhead_s"] * 1000.0, 5.1e-4, evidence, transport_path,
               fit["scope"])

    power_path = MANUSCRIPT / "figures" / "calibration_intervals.csv"
    power = pd.read_csv(power_path)
    power_mean = {
        (int(row.port), float(row.target_duty)): float(row.mean_w)
        for row in power.itertuples(index=False)
    }
    power_source_port = {1: 5, 3: 5, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8,
                         9: 9, 10: 9, 17: 9, 19: 5, 21: 9, 22: 9}
    for port, node in nodes.items():
        if port in (12, 20):
            for key in ("idle_w", "active_w"):
                record("power", port, key, node[key], node[key], 0.0,
                       "model assumption", args.deployment,
                       "No inline meter was available")
            continue
        if port == 23:
            expected_idle = sum(power_mean[(p, 0.0)] for p in (6, 7)) / 2
            expected_active = sum(power_mean[(p, 1.0)] for p in (6, 7)) / 2
            evidence = "same-class mean transfer"
            note = "mean of source ports 6 and 7"
        else:
            source_port = power_source_port[port]
            expected_idle = power_mean[(source_port, 0.0)]
            expected_active = power_mean[(source_port, 1.0)]
            evidence = "direct" if source_port == port else "same-class transfer"
            note = f"source port {source_port}"
        record("power", port, "idle_w", node["idle_w"], expected_idle,
               5.1e-5, evidence, power_path, note)
        record("power", port, "active_w", node["active_w"], expected_active,
               5.1e-5, evidence, power_path, note)

    thermal_path = PHYSICAL / "power_thermal_20260905" / "thermal_fit_analysis.json"
    thermal_fit = {int(row["port"]): row for row in load_json(thermal_path)["boards"]}
    thermal_source_port = {1: 5, 3: 5, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8,
                           9: 9, 10: 9, 17: 9, 19: 5, 21: 9, 22: 9}
    for port, node in nodes.items():
        if port in (12, 20):
            for key in ("idle_temperature_c", "thermal_gain_c_per_w", "thermal_tau_s"):
                record("thermal", port, key, node[key], node[key], 0.0,
                       "model assumption", args.deployment,
                       "Bracket around an observed temperature, not a fitted trace")
            continue
        if port == 23:
            expected_values = {
                "idle_temperature_c": sum(thermal_fit[p]["effective_idle_cpu_c"] for p in (6, 7)) / 2,
                "thermal_gain_c_per_w": sum(thermal_fit[p]["gain_c_per_pse_w"] for p in (6, 7)) / 2,
                "thermal_tau_s": sum(thermal_fit[p]["tau_s"] for p in (6, 7)) / 2,
            }
            evidence = "same-class mean transfer"
            note = "mean of source ports 6 and 7"
        else:
            source_port = thermal_source_port[port]
            fit = thermal_fit[source_port]
            expected_values = {
                "idle_temperature_c": fit["effective_idle_cpu_c"],
                "thermal_gain_c_per_w": fit["gain_c_per_pse_w"],
                "thermal_tau_s": fit["tau_s"],
            }
            evidence = "direct" if source_port == port else "same-class transfer"
            note = f"source port {source_port}"
        for key, expected in expected_values.items():
            record("thermal", port, key, node[key], expected, 5.1e-3,
                   evidence, thermal_path, note)

    startup_path = MANUSCRIPT / "figures" / "fig_container_startup_source_activations.csv"
    activation = pd.read_csv(startup_path)
    pivot = activation.pivot_table(
        index=["port", "block"], columns="method", values="ready_ms"
    )
    delta = ((pivot["cached_container"] - pivot["host_process"]) / 1000.0).rename("delta_s")
    startup = delta.groupby(level=0).agg(["mean", "std"])
    startup_source_port = {1: 5, 3: 5, 4: 5, 5: 5, 6: 7, 7: 7, 8: 8,
                           9: 9, 10: 9, 12: 9, 17: 9, 19: 5, 20: 9,
                           21: 9, 22: 9, 23: 7}
    for port, node in nodes.items():
        source_port = startup_source_port[port]
        evidence = "direct" if source_port == port else "same-class transfer"
        if port in (12, 20):
            evidence = "cross-platform sensitivity transfer"
        record("startup", port, "startup_mean_s", node["startup_mean_s"],
               startup.loc[source_port, "mean"], 5.1e-5, evidence, startup_path,
               f"paired cached-container minus host-process ready time, source port {source_port}")
        record("startup", port, "startup_sd_s", node["startup_sd_s"],
               startup.loc[source_port, "std"], 5.1e-5, evidence, startup_path,
               f"sample SD across paired blocks, source port {source_port}")

    for name, value in deployment["model_inputs"].items():
        if name.endswith("_evidence"):
            continue
        if name.startswith("payload_"):
            note = deployment["model_inputs"]["payload_evidence"]
        elif name.startswith("startup_"):
            note = deployment["model_inputs"]["startup_power_evidence"]
        elif name.startswith("fairness_"):
            note = deployment["model_inputs"]["fairness_evidence"]
        else:
            note = deployment["model_inputs"]["background_evidence"]
        record("explicit_model_input", 0, name, value, value, 0.0,
               "model assumption", args.deployment, str(note))

    controller = deployment["controller"]
    controller_model_inputs = (
        "cache_ttl_s",
        "cache_footprint_multiplier",
        "thermal_limit_c",
        "thermal_derate_start_c",
        "thermal_derate_slope_per_c",
        "thermal_minimum_service_multiplier",
        "energy_budget_fraction",
        "tenant_energy_share",
        "idle_allocation_fraction",
        "energy_burst_minutes",
        "admission_cap_per_tenant",
        "admission_window_s",
        "deadline_tail_multiplier",
        "service_prediction_quantile",
        "startup_prediction_quantile",
        "network_residual_exponent",
        "network_minimum_capacity_fraction",
        "minimum_deadline_s_by_tenant",
        "scd_v_joules",
        "simulated_inter_switch_downstream_ports",
        "simulated_inter_switch_propagation_ms",
    )
    for name in controller_model_inputs:
        value = controller[name]
        note = (
            controller["simulated_inter_switch_evidence"]
            if name.startswith("simulated_inter_switch")
            else "Explicit controller parameter fixed before the held-out test interval"
        )
        record(
            "controller_model_input",
            0,
            name,
            value,
            value,
            None if isinstance(value, list) else 0.0,
            "model assumption",
            args.deployment,
            note,
        )

    forecast_path = (
        args.deployment.parent.parent / Path(controller["service_forecast_source"])
    ).resolve()
    with np.load(forecast_path, allow_pickle=False) as forecast:
        prediction = np.asarray(forecast["prediction"])
        scale = np.asarray(forecast["normalization_scale"])
        origin = np.asarray(forecast["model_origin"])
    forecast_valid = bool(
        prediction.shape == (20160, 16)
        and scale.shape == prediction.shape
        and origin.shape == (20160,)
        and np.all(prediction >= 0)
        and np.all(scale > 0)
        and np.all((origin < 0) | (origin <= np.arange(20160)))
    )
    record(
        "causal_forecast",
        0,
        "shape_domain_and_past_only_origin",
        forecast_valid,
        True,
        None,
        "trained past-only artifact",
        forecast_path,
        controller["service_forecast_evidence"],
    )
    scheduler_path = ROOT / "src" / "scheduler_core.py"
    record(
        "policy_source",
        0,
        "fixed_manuscript_score_core_present",
        scheduler_path.is_file(),
        True,
        None,
        "source implementation",
        scheduler_path,
        controller["policy_parameter_status"],
    )

    frame = pd.DataFrame(rows)
    failures = frame[frame.status == "FAIL"]
    source_paths = sorted({Path(value) for value in frame.source if Path(value).is_file()})
    report = {
        "schema": "tnsm.strict-configuration-audit.v1",
        "deployment": str(args.deployment),
        "deployment_sha256": sha256(args.deployment),
        "checks": int(len(frame)),
        "passed": int((frame.status == "PASS").sum()),
        "failed": int(len(failures)),
        "evidence_class_counts": {
            str(key): int(value)
            for key, value in frame.evidence_class.value_counts().sort_index().items()
        },
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "status": "PASS" if failures.empty else "FAIL",
        "interpretation": (
            "PASS means the configured value matches its declared source or transfer rule. "
            "It does not convert transferred or model-assumption values into direct measurements."
        ),
        "failed_checks": failures.to_dict(orient="records"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "parameter_checks.csv", index=False)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
