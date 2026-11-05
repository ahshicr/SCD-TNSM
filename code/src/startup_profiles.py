from __future__ import annotations

"""Physical cold-start profile construction for the strict simulator.

The functions in this module contain no scheduling code.  They convert the
completed paired board measurements into auditable low, central, and high
lognormal launch-to-ready distributions, then map those distributions to every
node of a validated strict deployment.
"""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.strict_config import Deployment


PROFILE_NAMES = ("measured_low", "measured_central", "measured_high")
DIRECT_PORT_BY_CLASS = {"pi5": 5, "pi4": 9, "pi3": 7, "pi2": 8}
HOST_PORT_BY_CLASS = {"rdk_x5": 12, "jetson_nano": 20}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ci_half(record: dict[str, Any]) -> float:
    for name in ("halfwidth95", "ci95_half"):
        if name in record:
            return float(record[name])
    raise ValueError("measurement summary lacks a 95 percent CI half-width")


def _sample_sd(record: dict[str, Any]) -> float:
    n = int(record["n"])
    if n < 2:
        raise ValueError("at least two independent measurement blocks are required")
    return _ci_half(record) * math.sqrt(n) / float(stats.t.ppf(0.975, n - 1))


def _estimate(record: dict[str, Any]) -> dict[str, float | int]:
    mean = float(record["mean"])
    half = _ci_half(record)
    sd = _sample_sd(record)
    if not (math.isfinite(mean) and math.isfinite(half) and math.isfinite(sd)):
        raise ValueError("non-finite startup measurement")
    if mean <= 0 or half < 0 or sd < 0:
        raise ValueError("invalid startup measurement")
    return {"mean": mean, "ci95_half": half, "sd": sd, "n": int(record["n"])}


def _profile_record(mean: float, sd: float, basis: str) -> dict[str, float | str]:
    if mean <= 0 or sd < 0 or not math.isfinite(mean + sd):
        raise ValueError("invalid profile moment")
    return {"mean_s": mean / 1000.0, "sd_s": sd / 1000.0, "basis": basis}


def derive_startup_profiles(container_summary: Path, activation_summary: Path) -> dict[str, Any]:
    """Derive three class-level launch-to-ready distributions.

    Raspberry Pi profiles use direct cached-container measurements.  RDK X5
    and Jetson Nano combine their directly measured host activation with a
    Raspberry Pi container-orchestration surcharge.  The latter are therefore
    sensitivity brackets, not direct container measurements.
    """

    audit_path = container_summary.with_name("audit.json")
    if not audit_path.is_file():
        raise ValueError("completed container audit is required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_activations = 4 * 2 * 20
    if not audit.get("complete") or int(audit.get("activation_count", -1)) != expected_activations:
        raise ValueError("container campaign is incomplete")

    board_rows = json.loads(container_summary.read_text(encoding="utf-8"))
    boards = {int(row["port"]): row for row in board_rows}
    activation_payload = json.loads(activation_summary.read_text(encoding="utf-8"))
    activations = {int(row["port"]): row for row in activation_payload["activation"]}
    missing_direct = set(DIRECT_PORT_BY_CLASS.values()) - set(boards)
    missing_host = set(HOST_PORT_BY_CLASS.values()) - set(activations)
    if missing_direct or missing_host:
        raise ValueError(
            f"startup evidence is incomplete, direct={sorted(missing_direct)}, host={sorted(missing_host)}"
        )

    direct: dict[str, dict[str, float | int]] = {}
    surcharges: list[dict[str, float | int | str]] = []
    for device_class, port in DIRECT_PORT_BY_CLASS.items():
        metrics = boards[port]["metrics"]["ready_ms"]
        direct[device_class] = _estimate(metrics["cached_container"])
        surcharge = _estimate(metrics["paired_container_minus_host"])
        surcharges.append({**surcharge, "device_class": device_class, "port": port})

    low_surcharge = min(surcharges, key=lambda row: float(row["mean"]) - float(row["ci95_half"]))
    high_surcharge = max(surcharges, key=lambda row: float(row["mean"]) + float(row["ci95_half"]))
    central_surcharge_mean = float(np.median([float(row["mean"]) for row in surcharges]))
    central_surcharge_sd = float(np.median([float(row["sd"]) for row in surcharges]))
    surcharge_moments = {
        "measured_low": (
            max(1.0, float(low_surcharge["mean"]) - float(low_surcharge["ci95_half"])),
            float(low_surcharge["sd"]),
            f"minimum lower 95 percent bound across direct Pi pairs, port {low_surcharge['port']}",
        ),
        "measured_central": (
            central_surcharge_mean,
            central_surcharge_sd,
            "median direct Pi paired surcharge and median sample standard deviation",
        ),
        "measured_high": (
            float(high_surcharge["mean"]) + float(high_surcharge["ci95_half"]),
            float(high_surcharge["sd"]),
            f"maximum upper 95 percent bound across direct Pi pairs, port {high_surcharge['port']}",
        ),
    }

    profiles: dict[str, dict[str, dict[str, float | str]]] = {
        name: {} for name in PROFILE_NAMES
    }
    for device_class, estimate in direct.items():
        central_mean = float(estimate["mean"])
        central_sd = float(estimate["sd"])
        means = {
            "measured_low": max(1.0, central_mean - float(estimate["ci95_half"])),
            "measured_central": central_mean,
            "measured_high": central_mean + float(estimate["ci95_half"]),
        }
        for name, mean in means.items():
            scaled_sd = central_sd * mean / central_mean
            profiles[name][device_class] = _profile_record(
                mean,
                scaled_sd,
                f"direct cached-container port {DIRECT_PORT_BY_CLASS[device_class]}, {name}",
            )

    for device_class, port in HOST_PORT_BY_CLASS.items():
        host = _estimate(activations[port]["worker_activation_s"])
        host_mean_ms = float(host["mean"]) * 1000.0
        host_sd_ms = float(host["sd"]) * 1000.0
        for name in PROFILE_NAMES:
            surcharge_mean, surcharge_sd, surcharge_basis = surcharge_moments[name]
            profiles[name][device_class] = _profile_record(
                host_mean_ms + surcharge_mean,
                math.sqrt(host_sd_ms * host_sd_ms + surcharge_sd * surcharge_sd),
                f"measured host activation port {port} plus {surcharge_basis}",
            )

    expected_classes = set(DIRECT_PORT_BY_CLASS) | set(HOST_PORT_BY_CLASS)
    for name, profile in profiles.items():
        if set(profile) != expected_classes:
            raise ValueError(f"profile {name} does not cover every device class")
        for device_class in expected_classes:
            if profile[device_class]["mean_s"] <= 0 or profile[device_class]["sd_s"] < 0:
                raise ValueError(f"invalid {name} moments for {device_class}")
    for device_class in expected_classes:
        means = [float(profiles[name][device_class]["mean_s"]) for name in PROFILE_NAMES]
        if not means[0] <= means[1] <= means[2]:
            raise ValueError(f"profile ordering failed for {device_class}: {means}")

    evidence = {
        "schema": "tnsm.strict-startup-profiles.v1",
        "profiles": profiles,
        "direct_container_ports": DIRECT_PORT_BY_CLASS,
        "host_activation_ports": HOST_PORT_BY_CLASS,
        "non_pi_boundary": "model-derived sensitivity bracket, not a direct container measurement",
        "timing_boundary": (
            "cached launch-to-ready on an authenticated transport, excluding image acquisition and "
            "including runtime launch, model transfer, initialization, and readiness"
        ),
        "container_summary_sha256": sha256(container_summary),
        "container_audit_sha256": sha256(audit_path),
        "activation_summary_sha256": sha256(activation_summary),
    }
    evidence["profile_sha256"] = canonical_hash(evidence)
    return evidence


def _lognormal_moments(mean: float, sd: float) -> tuple[float, float]:
    if mean <= 0 or sd < 0:
        raise ValueError("lognormal moments must be nonnegative with positive mean")
    ratio = (sd / mean) ** 2
    sigma = math.sqrt(math.log1p(ratio))
    return math.log(mean) - 0.5 * sigma * sigma, sigma


def deployment_with_startup_profile(
    deployment: Deployment,
    evidence: dict[str, Any],
    profile_name: str,
    effective_source_sha256: str,
) -> Deployment:
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unknown startup profile {profile_name}")
    profile = evidence["profiles"][profile_name]
    mu = np.empty(deployment.nodes, dtype=np.float64)
    sigma = np.empty(deployment.nodes, dtype=np.float64)
    for index, raw_class in enumerate(deployment.device_class):
        device_class = str(raw_class)
        if device_class not in profile:
            raise ValueError(f"startup profile does not cover {device_class}")
        moments = profile[device_class]
        mu[index], sigma[index] = _lognormal_moments(
            float(moments["mean_s"]), float(moments["sd_s"])
        )
    return replace(
        deployment,
        source_sha256=effective_source_sha256,
        startup_mu_log_s=mu,
        startup_sigma_log_s=sigma,
    )
