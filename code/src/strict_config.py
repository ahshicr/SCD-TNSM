from __future__ import annotations

"""Validated configuration loader for the request-level simulator."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEPLOYMENT_SCHEMA = "tnsm.strict-deployment.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _lognormal_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    if mean <= 0 or sd < 0:
        raise ValueError("lognormal mean must be positive and sd nonnegative")
    variance_ratio = (sd / mean) ** 2
    sigma = math.sqrt(math.log1p(variance_ratio))
    mu = math.log(mean) - 0.5 * sigma * sigma
    return mu, sigma


@dataclass(frozen=True)
class Deployment:
    source_path: Path
    source_sha256: str
    raw: dict[str, object]
    node_id: np.ndarray
    port: np.ndarray
    device_class: np.ndarray
    capability_mask: np.ndarray
    usable_memory_mb: np.ndarray
    service_rate: np.ndarray
    service_rate_profiles: dict[str, np.ndarray]
    idle_w: np.ndarray
    active_w: np.ndarray
    incremental_w: np.ndarray
    energy_budget_j_per_s: np.ndarray
    tenant_energy_share: np.ndarray
    idle_temperature_c: np.ndarray
    thermal_gain_c_per_w: np.ndarray
    thermal_tau_s: np.ndarray
    thermal_limit_c: float
    thermal_derate_start_c: float
    thermal_derate_slope_per_c: float
    thermal_minimum_service_multiplier: float
    startup_mu_log_s: np.ndarray
    startup_sigma_log_s: np.ndarray
    route_links: np.ndarray
    route_length: np.ndarray
    link_rate_bps: np.ndarray
    link_fixed_s: np.ndarray
    link_propagation_s: np.ndarray
    link_background_fraction: np.ndarray
    network_residual_exponent: float
    network_minimum_capacity_fraction: float
    cache_ttl_s: float
    cache_footprint_multiplier: float
    admission_cap_per_tenant: int
    admission_window_s: float
    deadline_tail_multiplier: float
    service_prediction_quantile: float
    startup_prediction_quantile: float
    scd_v_joules: float
    service_forecast_path: Path
    service_forecast_evidence: str
    simulated_inter_switch_downstream_ports: np.ndarray
    simulated_inter_switch_propagation_s: float
    simulated_inter_switch_evidence: str
    idle_allocation_fraction: float
    payload_minimum_bytes: float
    payload_sqrt_memory_coefficient_bytes: float
    startup_incremental_power_fraction: float
    background_ar1_coefficient: float
    background_innovation_sd: float
    background_periodic_amplitude: float
    background_period_minutes: int
    background_burst_period_minutes: int
    background_burst_start_minute: int
    background_burst_duration_minutes: int
    background_burst_amplitude: float
    background_forecast_alpha: float
    background_maximum_fraction: float
    minimum_deadline_s_by_tenant: np.ndarray

    @property
    def nodes(self) -> int:
        return int(self.node_id.size)

    @property
    def links(self) -> int:
        return int(self.link_rate_bps.size)


def load_deployment(path: Path, *, include_sensitivity_nodes: bool = False) -> Deployment:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != DEPLOYMENT_SCHEMA:
        raise ValueError(f"unsupported deployment schema: {raw.get('schema')}")
    links = sorted(raw["links"], key=lambda item: item["id"])
    if [item["id"] for item in links] != list(range(len(links))):
        raise ValueError("link ids must be unique and contiguous from zero")
    excluded = set() if include_sensitivity_nodes else set(raw["primary_excluded_ports"])
    nodes = [item for item in raw["nodes"] if item["port"] not in excluded]
    nodes.sort(key=lambda item: item["id"])
    if not include_sensitivity_nodes and len(nodes) != 15:
        raise ValueError("the frozen primary deployment must contain exactly 15 nodes")
    ports = [int(item["port"]) for item in nodes]
    if len(ports) != len(set(ports)):
        raise ValueError("node ports must be unique")
    for node in nodes:
        if not node["route"] or node["route"][0] != 0:
            raise ValueError(f"node on port {node['port']} lacks the shared ingress route")
        if any(link < 0 or link >= len(links) for link in node["route"]):
            raise ValueError(f"node on port {node['port']} references an invalid link")
        if node["usable_memory_mb"] > node["memory_total_mb"]:
            raise ValueError(f"usable memory exceeds physical memory on port {node['port']}")
        for evidence_key in ("power_evidence", "thermal_evidence", "startup_evidence"):
            if not str(node.get(evidence_key, "")).strip():
                raise ValueError(f"missing {evidence_key} on port {node['port']}")
    service_rate_profiles: dict[str, np.ndarray] = {
        "batch16": np.asarray([item["service_rate"] for item in nodes], dtype=np.float64)
    }
    for profile_name, profile in raw.get("service_rate_profiles", {}).items():
        missing_ports = [port for port in ports if str(port) not in profile]
        if missing_ports:
            raise ValueError(
                f"service profile {profile_name} is missing ports {missing_ports}"
            )
        values = np.asarray([profile[str(port)] for port in ports], dtype=np.float64)
        if np.any(values <= 0):
            raise ValueError(f"service profile {profile_name} contains nonpositive rates")
        service_rate_profiles[str(profile_name)] = values

    max_hops = max(len(item["route"]) for item in nodes)
    route_links = np.full((len(nodes), max_hops), -1, dtype=np.int16)
    route_length = np.zeros(len(nodes), dtype=np.int8)
    for index, node in enumerate(nodes):
        route = np.asarray(node["route"], dtype=np.int16)
        route_links[index, : len(route)] = route
        route_length[index] = len(route)

    startup = [_lognormal_from_mean_sd(item["startup_mean_s"], item["startup_sd_s"]) for item in nodes]
    controller = raw["controller"]
    minimum_deadline = np.asarray(
        controller["minimum_deadline_s_by_tenant"], dtype=np.float64
    )
    if minimum_deadline.shape != (4,) or np.any(minimum_deadline <= 0):
        raise ValueError("four positive tenant deadline floors are required")
    tenant_energy_share = np.asarray(
        controller["tenant_energy_share"], dtype=np.float64
    )
    if (
        tenant_energy_share.shape != (4,)
        or np.any(tenant_energy_share <= 0)
        or not math.isclose(float(tenant_energy_share.sum()), 1.0, abs_tol=1.0e-12)
    ):
        raise ValueError("four positive tenant energy shares summing to one are required")
    for name in ("service_prediction_quantile", "startup_prediction_quantile"):
        if not 0.5 <= float(controller[name]) < 1.0:
            raise ValueError(f"{name} must lie in [0.5, 1)")
    if not 0 <= float(controller["idle_allocation_fraction"]) <= 1:
        raise ValueError("idle_allocation_fraction must lie in [0, 1]")
    if float(controller["thermal_derate_slope_per_c"]) < 0:
        raise ValueError("thermal derating slope must be nonnegative")
    if not 0 < float(controller["thermal_minimum_service_multiplier"]) <= 1:
        raise ValueError("minimum thermal service multiplier must lie in (0, 1]")
    if float(controller["thermal_derate_start_c"]) > float(controller["thermal_limit_c"]):
        raise ValueError("thermal derating must begin no later than the safety limit")
    if float(controller["network_residual_exponent"]) <= 0:
        raise ValueError("network residual exponent must be positive")
    if not 0 < float(controller["network_minimum_capacity_fraction"]) <= 1:
        raise ValueError("minimum network capacity fraction must lie in (0, 1]")
    admission_cap = int(controller["admission_cap_per_tenant"])
    if admission_cap <= 0 or admission_cap != float(controller["admission_cap_per_tenant"]):
        raise ValueError("admission_cap_per_tenant must be a positive integer")
    if float(controller["admission_window_s"]) <= 0:
        raise ValueError("admission_window_s must be positive")
    if float(controller["scd_v_joules"]) <= 0:
        raise ValueError("scd_v_joules must be positive")
    forecast_rel = Path(str(controller["service_forecast_source"]))
    forecast_path = (path.parent.parent / forecast_rel).resolve()
    if not forecast_path.is_file():
        raise ValueError(f"service forecast does not exist: {forecast_path}")
    forecast_evidence = str(controller.get("service_forecast_evidence", "")).strip()
    if not forecast_evidence:
        raise ValueError("service_forecast_evidence is required")
    downstream_ports = np.asarray(
        controller["simulated_inter_switch_downstream_ports"], dtype=np.int16
    )
    if downstream_ports.ndim != 1 or downstream_ports.size == 0:
        raise ValueError("simulated inter-switch topology needs downstream ports")
    if any(int(port) not in ports for port in downstream_ports):
        raise ValueError("simulated inter-switch topology references an absent port")
    inter_switch_propagation_s = (
        float(controller["simulated_inter_switch_propagation_ms"]) / 1000.0
    )
    if inter_switch_propagation_s < 0:
        raise ValueError("simulated inter-switch propagation must be nonnegative")
    inter_switch_evidence = str(
        controller.get("simulated_inter_switch_evidence", "")
    ).strip()
    if not inter_switch_evidence:
        raise ValueError("simulated_inter_switch_evidence is required")
    model = raw["model_inputs"]
    for name in (
        "payload_minimum_bytes",
        "payload_sqrt_memory_coefficient_bytes",
        "background_period_minutes",
        "background_burst_period_minutes",
    ):
        if float(model[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= float(model["startup_incremental_power_fraction"]) <= 1:
        raise ValueError("startup_incremental_power_fraction must lie in [0, 1]")
    if not -1 < float(model["background_ar1_coefficient"]) < 1:
        raise ValueError("background_ar1_coefficient must lie in (-1, 1)")
    if float(model["background_innovation_sd"]) < 0:
        raise ValueError("background_innovation_sd must be nonnegative")
    if not 0 < float(model["background_forecast_alpha"]) <= 1:
        raise ValueError("background_forecast_alpha must lie in (0, 1]")
    if not 0 < float(model["background_maximum_fraction"]) < 1:
        raise ValueError("background_maximum_fraction must lie in (0, 1)")
    burst_start = int(model["background_burst_start_minute"])
    burst_duration = int(model["background_burst_duration_minutes"])
    burst_period = int(model["background_burst_period_minutes"])
    if burst_start < 0 or burst_duration <= 0 or burst_start + burst_duration > burst_period:
        raise ValueError("background burst interval must lie inside its positive period")

    active = np.asarray([item["active_w"] for item in nodes], dtype=np.float64)
    idle = np.asarray([item["idle_w"] for item in nodes], dtype=np.float64)
    incremental = active - idle
    if np.any(incremental <= 0):
        raise ValueError("active power must exceed idle power on every node")
    budget_fraction = float(controller["energy_budget_fraction"])
    if not 0 < budget_fraction <= 1:
        raise ValueError("energy budget fraction must lie in (0, 1]")

    return Deployment(
        source_path=path,
        source_sha256=_sha256(path),
        raw=raw,
        node_id=np.asarray([item["id"] for item in nodes], dtype=np.int16),
        port=np.asarray(ports, dtype=np.int16),
        device_class=np.asarray([item["device_class"] for item in nodes], dtype="U32"),
        capability_mask=np.asarray([item["capability_mask"] for item in nodes], dtype=np.int32),
        usable_memory_mb=np.asarray([item["usable_memory_mb"] for item in nodes], dtype=np.float64),
        service_rate=np.asarray([item["service_rate"] for item in nodes], dtype=np.float64),
        service_rate_profiles=service_rate_profiles,
        idle_w=idle,
        active_w=active,
        incremental_w=incremental,
        energy_budget_j_per_s=incremental * budget_fraction,
        tenant_energy_share=tenant_energy_share,
        idle_temperature_c=np.asarray([item["idle_temperature_c"] for item in nodes], dtype=np.float64),
        thermal_gain_c_per_w=np.asarray([item["thermal_gain_c_per_w"] for item in nodes], dtype=np.float64),
        thermal_tau_s=np.asarray([item["thermal_tau_s"] for item in nodes], dtype=np.float64),
        thermal_limit_c=float(controller["thermal_limit_c"]),
        thermal_derate_start_c=float(controller["thermal_derate_start_c"]),
        thermal_derate_slope_per_c=float(controller["thermal_derate_slope_per_c"]),
        thermal_minimum_service_multiplier=float(
            controller["thermal_minimum_service_multiplier"]
        ),
        startup_mu_log_s=np.asarray([item[0] for item in startup], dtype=np.float64),
        startup_sigma_log_s=np.asarray([item[1] for item in startup], dtype=np.float64),
        route_links=route_links,
        route_length=route_length,
        link_rate_bps=np.asarray([item["rate_mbps"] * 1.0e6 for item in links], dtype=np.float64),
        link_fixed_s=np.asarray([item.get("fixed_overhead_ms", 0.0) / 1000.0 for item in links], dtype=np.float64),
        link_propagation_s=np.asarray([item["propagation_ms"] / 1000.0 for item in links], dtype=np.float64),
        link_background_fraction=np.asarray([item["nominal_background_fraction"] for item in links], dtype=np.float64),
        network_residual_exponent=float(controller["network_residual_exponent"]),
        network_minimum_capacity_fraction=float(
            controller["network_minimum_capacity_fraction"]
        ),
        cache_ttl_s=float(controller["cache_ttl_s"]),
        cache_footprint_multiplier=float(controller["cache_footprint_multiplier"]),
        admission_cap_per_tenant=admission_cap,
        admission_window_s=float(controller["admission_window_s"]),
        deadline_tail_multiplier=float(controller["deadline_tail_multiplier"]),
        service_prediction_quantile=float(controller["service_prediction_quantile"]),
        startup_prediction_quantile=float(controller["startup_prediction_quantile"]),
        scd_v_joules=float(controller["scd_v_joules"]),
        service_forecast_path=forecast_path,
        service_forecast_evidence=forecast_evidence,
        simulated_inter_switch_downstream_ports=downstream_ports,
        simulated_inter_switch_propagation_s=inter_switch_propagation_s,
        simulated_inter_switch_evidence=inter_switch_evidence,
        idle_allocation_fraction=float(controller["idle_allocation_fraction"]),
        payload_minimum_bytes=float(model["payload_minimum_bytes"]),
        payload_sqrt_memory_coefficient_bytes=float(
            model["payload_sqrt_memory_coefficient_bytes"]
        ),
        startup_incremental_power_fraction=float(
            model["startup_incremental_power_fraction"]
        ),
        background_ar1_coefficient=float(model["background_ar1_coefficient"]),
        background_innovation_sd=float(model["background_innovation_sd"]),
        background_periodic_amplitude=float(model["background_periodic_amplitude"]),
        background_period_minutes=int(model["background_period_minutes"]),
        background_burst_period_minutes=burst_period,
        background_burst_start_minute=burst_start,
        background_burst_duration_minutes=burst_duration,
        background_burst_amplitude=float(model["background_burst_amplitude"]),
        background_forecast_alpha=float(model["background_forecast_alpha"]),
        background_maximum_fraction=float(model["background_maximum_fraction"]),
        minimum_deadline_s_by_tenant=minimum_deadline,
    )
