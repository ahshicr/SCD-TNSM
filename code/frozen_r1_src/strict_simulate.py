from __future__ import annotations

"""Request-level discrete-event simulator for the strict revision.

The policy state and realised execution state are deliberately separate.  The
policy sees medians, causal link-load forecasts and committed energy.  Realised
service, transport, startup, cache and thermal states are never copied back into
the policy before their completion event is processed.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import numba as nb
import numpy as np

from src.scheduler_core import candidate_score as scd_candidate_score
from src.scheduler_core import hop_jitter_mean
from src.scheduler_core import operating_penalty
from src.strict_config import Deployment, load_deployment
from src.strict_trace import RequestTrace, load_request_trace


METHODS = {
    # The SCD-family identifiers deliberately match scheduler_core.py.  This
    # prevents the simulator and physical adapter from drifting into separate
    # implementations of the manuscript's policy.
    "scd": 10,
    "compute_first": 1,
    "energy_first": 2,
    "dpp_matched": 5,
    "round_robin": 4,
    "scd_no_debt": 6,
    "scd_no_network": 7,
    "scd_no_forecast": 8,
    "scd_static_weights": 9,
    "scd_no_thermal_term": 11,
    "scd_independent_network": 14,
    "scd_shared_energy": 13,
}
@dataclass(frozen=True)
class StrictScenario:
    name: str = "nominal"
    seed: int = 20260914
    background_multiplier: float = 1.0
    inter_switch_rate_mbps: float = 0.0
    service_variability_multiplier: float = 1.0
    service_rate_profile: str = "batch16"
    startup_variability_multiplier: float = 1.0
    power_coefficient_of_variation: float = 0.15
    power_margin_enabled: bool = True
    thermal_gain_multiplier: float = 1.0
    thermal_derate_multiplier: float = 1.0
    cache_ttl_s: float | None = None
    cache_footprint_multiplier: float = 1.0
    cache_identity_mode: str = "function"
    cache_slot_cap: int = 0
    payload_multiplier: float = 1.0
    energy_budget_factor: float = 1.0
    energy_charge_margin: float = 1.0
    admission_cap_factor: float = 1.0
    admission_guard: bool = True
    thermal_guard: bool = True
    energy_guard: bool = True
    deadline_guard: bool = False
    deadline_guard_multiplier: float = 4.0
    deadline_floor_multiplier: float = 1.0
    deadline_tail_multiplier_factor: float = 1.0
    accelerator_fraction: float = 0.0
    include_sensitivity_nodes: bool = False
    attack_tenant: int = 3
    attack_multiplier: int = 1
    attack_jitter_s: float = 0.010
    start_day: int = 0
    end_day_exclusive: int = 0
    max_requests: int = 0


@dataclass(frozen=True)
class StrictResult:
    method: str
    scenario: str
    seed: int
    trace_requests: int
    evaluated_requests: int
    evaluation_start_s: float
    evaluation_end_s: float
    admitted_requests: int
    rejected_requests: int
    on_time_requests: int
    success_ratio: float
    miss_ratio: float
    deadline_ceiling_success_ratio: float
    deadline_ceiling_miss_ratio: float
    deadline_ceiling_on_time_requests: int
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    task_energy_mj: float
    supplied_energy_mj: float
    mean_peak_aoi_ms: float
    p95_peak_aoi_ms: float
    jain_tenant_success: float
    minimum_tenant_success_ratio: float
    legitimate_success_ratio: float
    legitimate_offered_requests: int
    legitimate_on_time_requests: int
    attacker_offered_requests: int
    attacker_admitted_requests: int
    attacker_admission_ratio: float
    actual_cold_start_ratio: float
    predicted_cold_start_ratio: float
    actual_cold_start_requests: int
    predicted_cold_start_requests: int
    false_warm_ratio: float
    false_cold_ratio: float
    false_warm_requests: int
    false_cold_requests: int
    predicted_cache_evictions: int
    actual_cache_evictions: int
    imputed_duration_request_ratio: float
    cache_identity_mode: str
    cache_slots_allocated: int
    admission_cap_per_tenant: int
    admission_window_s: float
    admission_guard: bool
    deadline_guard: bool
    deadline_ceiling_multiplier: float
    inter_switch_rate_mbps: float
    maximum_route_hops: int
    rate_rejections: int
    energy_rejections: int
    compatibility_rejections: int
    thermal_rejections: int
    deadline_rejections: int
    p95_network_queue_ms: float
    p95_network_jitter_ms: float
    peak_temperature_c: float
    thermal_violations: int
    thermal_derate_ratio: float
    thermal_derated_requests: int
    energy_underestimation_ratio: float
    energy_underestimated_requests: int
    energy_shortfall_j: float
    maximum_per_device_energy_debt_j: float
    final_total_device_energy_debt_j: float
    maximum_per_device_tenant_energy_debt_j: float
    final_total_tenant_energy_debt_j: float
    last_completion_s: float
    drain_duration_s: float
    peak_aoi_samples: int
    runtime_s: float
    deployment_sha256: str
    simulator_sha256: str
    scheduler_core_sha256: str
    service_forecast_sha256: str
    trace_manifest_sha256: str
    tenant_offered: list[int]
    tenant_admitted: list[int]
    tenant_success: list[int]
    node_ports: list[int]
    node_assignments: list[int]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def required_cache_slots(
    memory_mb: np.ndarray,
    usable_memory_mb: np.ndarray,
    footprint_multiplier: float,
    cache_identity: np.ndarray | None = None,
    slot_cap: int = 0,
) -> int:
    """Return a provable upper bound on simultaneously resident services.

    One slot is needed per resident function identity.  The bound is derived
    from the largest usable device memory and the smallest positive container
    footprint in the evaluated scenario.  Consequently the array dimension
    cannot silently evict an entry before the explicit memory/TTL LRU policy
    does.
    """

    if footprint_multiplier <= 0:
        raise ValueError("cache footprint multiplier must be positive")
    footprints = np.asarray(memory_mb, dtype=np.float64) * footprint_multiplier
    positive = footprints[footprints > 0]
    if positive.size != footprints.size:
        raise ValueError("every cache footprint must be positive")
    bound = int(math.floor(float(np.max(usable_memory_mb)) / float(np.min(positive))))
    identities = int(footprints.size)
    if cache_identity is not None:
        identities = int(np.unique(np.asarray(cache_identity)).size)
    if slot_cap < 0:
        raise ValueError("cache slot cap must be nonnegative")
    cardinality = identities if slot_cap == 0 else min(identities, slot_cap)
    return max(1, min(cardinality, bound))


def build_cache_model(
    trace: RequestTrace, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build the cache identity and internally consistent footprint mapping."""

    memory = np.asarray(trace.memory_mb, dtype=np.float64)
    if mode == "function":
        return np.arange(memory.size, dtype=np.int32), memory
    if mode != "stratum":
        raise ValueError("cache_identity_mode must be 'function' or 'stratum'")
    identity = np.asarray(trace.stratum_id, dtype=np.int32)
    footprint_by_stratum = np.zeros(int(identity.max()) + 1, dtype=np.float64)
    for stratum in range(footprint_by_stratum.size):
        members = memory[identity == stratum]
        if members.size:
            footprint_by_stratum[stratum] = float(np.max(members))
    return identity, footprint_by_stratum[identity]


def build_scenario_topology(
    deployment: Deployment, scenario: StrictScenario
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the physical one-switch paths or an explicit two-switch sensitivity.

    The additional inter-switch resource is never presented as measured H3C
    behavior.  It is inserted between the shared ingress and the egress links
    of the configured downstream-port group, which makes FCFS coupling across
    those routes explicit.
    """

    if scenario.inter_switch_rate_mbps < 0:
        raise ValueError("inter_switch_rate_mbps must be nonnegative")
    if scenario.inter_switch_rate_mbps == 0:
        return (
            deployment.route_links,
            deployment.route_length,
            deployment.link_rate_bps,
            deployment.link_fixed_s,
            deployment.link_propagation_s,
            deployment.link_background_fraction,
        )
    new_link = len(deployment.link_rate_bps)
    routes = np.full((deployment.nodes, 3), -1, dtype=np.int16)
    lengths = deployment.route_length.copy()
    downstream = set(int(value) for value in deployment.simulated_inter_switch_downstream_ports)
    for node in range(deployment.nodes):
        original = deployment.route_links[node, : deployment.route_length[node]]
        if int(deployment.port[node]) in downstream:
            routes[node, 0] = original[0]
            routes[node, 1] = new_link
            routes[node, 2] = original[-1]
            lengths[node] = 3
        else:
            routes[node, : len(original)] = original
    return (
        routes,
        lengths,
        np.append(deployment.link_rate_bps, scenario.inter_switch_rate_mbps * 1.0e6),
        np.append(deployment.link_fixed_s, 0.0),
        np.append(
            deployment.link_propagation_s,
            deployment.simulated_inter_switch_propagation_s,
        ),
        np.append(
            deployment.link_background_fraction,
            deployment.link_background_fraction[0],
        ),
    )


def build_background_load(
    deployment: Deployment,
    minutes: int,
    scenario: StrictScenario,
    link_background_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a paired exogenous load process and a strictly causal EWMA forecast."""

    configured_background = (
        deployment.link_background_fraction
        if link_background_fraction is None
        else np.asarray(link_background_fraction, dtype=np.float64)
    )
    base = np.clip(
        configured_background * scenario.background_multiplier,
        0.0,
        deployment.background_maximum_fraction,
    )
    actual = np.empty((minutes, len(configured_background)), dtype=np.float32)
    for link in range(len(configured_background)):
        # One named stream per link preserves every existing exogenous series
        # when a sensitivity topology appends another resource.
        rng = np.random.Generator(
            np.random.PCG64(scenario.seed ^ 0x5A17C9E3 ^ ((link + 1) * 0x9E3779B1))
        )
        innovation = 0.0
        for minute in range(minutes):
            innovation = (
                deployment.background_ar1_coefficient * innovation
                + rng.normal(0.0, deployment.background_innovation_sd)
            )
            phase = deployment.background_periodic_amplitude * math.sin(
                2.0 * math.pi * minute / deployment.background_period_minutes
            )
            burst = 0.0
            within_period = minute % deployment.background_burst_period_minutes
            if (
                deployment.background_burst_start_minute
                <= within_period
                < deployment.background_burst_start_minute
                + deployment.background_burst_duration_minutes
            ):
                burst = deployment.background_burst_amplitude * scenario.background_multiplier
            actual[minute, link] = np.clip(
                base[link] + innovation + phase + burst,
                0.0,
                deployment.background_maximum_fraction,
            )
    forecast = np.empty_like(actual)
    forecast[0] = base
    for minute in range(1, minutes):
        alpha = deployment.background_forecast_alpha
        forecast[minute] = (1.0 - alpha) * forecast[minute - 1] + alpha * actual[minute - 1]
    return forecast, actual


def load_service_pressure(path: Path, minutes: int) -> tuple[np.ndarray, str]:
    """Load the causal 16-class forecast and return manuscript-scale pressure.

    The scheduler uses ``log(1 + forecast / past-only scale)``.  The exported
    ``model_origin`` is checked here so a stale or future-fitted artifact cannot
    silently enter a strict run.
    """

    with np.load(path, allow_pickle=False) as data:
        required = {"prediction", "normalization_scale", "model_origin"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"service forecast is missing arrays: {sorted(missing)}")
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        scale = np.asarray(data["normalization_scale"], dtype=np.float64)
        origin = np.asarray(data["model_origin"], dtype=np.int64)
    if prediction.shape != scale.shape or prediction.ndim != 2 or prediction.shape[1] != 16:
        raise ValueError("service forecast must contain matching [minute,16] arrays")
    if origin.shape != (prediction.shape[0],):
        raise ValueError("model_origin must contain one entry per minute")
    minute_index = np.arange(prediction.shape[0], dtype=np.int64)
    if np.any((origin >= 0) & (origin > minute_index)):
        raise ValueError("service forecast contains a future model origin")
    if np.any(prediction < 0) or np.any(scale <= 0):
        raise ValueError("service forecast prediction and scale are outside their domains")
    pressure = np.log1p(prediction / scale).astype(np.float32)
    if minutes > pressure.shape[0]:
        raise ValueError(
            f"service forecast has {pressure.shape[0]} minutes but the run needs {minutes}"
        )
    return np.ascontiguousarray(pressure[:minutes]), _file_sha256(path)


def build_power_factors(
    minutes: int,
    node_ids: int | np.ndarray,
    scenario: StrictScenario,
) -> tuple[float, np.ndarray]:
    """Return a marginal p95 predictor and paired mean-one realized factors."""

    identifiers = (
        np.arange(node_ids, dtype=np.int64)
        if isinstance(node_ids, int)
        else np.asarray(node_ids, dtype=np.int64)
    )
    nodes = len(identifiers)
    coefficient = scenario.power_coefficient_of_variation
    if coefficient < 0:
        raise ValueError("power coefficient of variation must be nonnegative")
    if coefficient == 0:
        return 1.0, np.ones((minutes, nodes), dtype=np.float32)
    eta = math.sqrt(math.log1p(coefficient * coefficient))
    predicted = (
        math.exp(NormalDist().inv_cdf(0.95) * eta - 0.5 * eta * eta)
        if scenario.power_margin_enabled
        else 1.0
    )
    actual = np.empty((minutes, nodes), dtype=np.float32)
    for column, identifier in enumerate(identifiers):
        rng = np.random.Generator(
            np.random.PCG64(
                scenario.seed ^ 0x37C4D25B ^ ((int(identifier) + 1) * 0x85EBCA77)
            )
        )
        actual[:, column] = rng.lognormal(
            mean=-0.5 * eta * eta,
            sigma=eta,
            size=minutes,
        ).astype(np.float32)
    return predicted, actual


def build_requirement_mask(trace: RequestTrace, scenario: StrictScenario) -> np.ndarray:
    requirement = np.ones(len(trace.memory_mb), dtype=np.int32)
    fraction = scenario.accelerator_fraction
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("accelerator_fraction must lie in [0, 1]")
    if fraction == 0:
        return requirement
    digest = trace.function_key_digest.astype(np.uint64, copy=False)
    threshold = int(fraction * 10_000)
    selected = (digest % np.uint64(10_000)) < np.uint64(threshold)
    use_gpu = ((digest >> np.uint64(14)) & np.uint64(1)) == 0
    requirement[selected & use_gpu] |= 8
    requirement[selected & ~use_gpu] |= 16
    return requirement


def augment_with_attack(
    release_s: np.ndarray,
    function_id: np.ndarray,
    service_z: np.ndarray,
    network_z: np.ndarray,
    startup_z: np.ndarray,
    tenant_by_function: np.ndarray,
    scenario: StrictScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add reproducible synthetic flood requests without changing legitimate jobs."""

    multiplier = scenario.attack_multiplier
    if multiplier < 1 or int(multiplier) != multiplier:
        raise ValueError("attack_multiplier must be a positive integer")
    if not 0 <= scenario.attack_tenant < 4:
        raise ValueError("attack_tenant must lie in [0, 3]")
    if scenario.attack_jitter_s < 0:
        raise ValueError("attack_jitter_s must be nonnegative")
    attack_start_s = scenario.start_day * 86400.0
    attack_stop_s = (
        scenario.end_day_exclusive * 86400.0
        if scenario.end_day_exclusive
        else np.inf
    )
    base_attack = (
        (tenant_by_function[function_id] == scenario.attack_tenant)
        & (release_s >= attack_start_s)
        & (release_s < attack_stop_s)
    )
    is_attack = np.zeros(len(release_s), dtype=np.uint8)
    if multiplier == 1 or not np.any(base_attack):
        return release_s, function_id, service_z, network_z, startup_z, is_attack

    repeats = multiplier - 1
    attack_release = np.repeat(release_s[base_attack], repeats)
    offsets = np.tile(np.arange(1, multiplier, dtype=np.float64), int(base_attack.sum()))
    attack_release += offsets * (scenario.attack_jitter_s / multiplier)
    attack_function = np.repeat(function_id[base_attack], repeats)
    cloned = len(attack_release)
    rng_service = np.random.Generator(np.random.PCG64(scenario.seed ^ 0xA71C011))
    rng_network = np.random.Generator(np.random.PCG64(scenario.seed ^ 0xA71C022))
    rng_startup = np.random.Generator(np.random.PCG64(scenario.seed ^ 0xA71C033))
    combined_release = np.concatenate((release_s, attack_release))
    combined_function = np.concatenate((function_id, attack_function))
    combined_service = np.concatenate(
        (service_z, rng_service.standard_normal(cloned).astype(np.float32))
    )
    combined_network = np.concatenate(
        (
            network_z,
            rng_network.standard_normal((cloned, network_z.shape[1])).astype(np.float32),
        ),
        axis=0,
    )
    combined_startup = np.concatenate(
        (startup_z, rng_startup.standard_normal(cloned).astype(np.float32))
    )
    combined_attack = np.concatenate(
        (is_attack, np.ones(cloned, dtype=np.uint8))
    )
    order = np.argsort(combined_release, kind="stable")
    return (
        combined_release[order],
        combined_function[order],
        combined_service[order],
        combined_network[order],
        combined_startup[order],
        combined_attack[order],
    )


@nb.njit(cache=True)
def _cache_status(
    service: int,
    footprint_mb: float,
    start_s: float,
    service_slots: np.ndarray,
    last_completion: np.ndarray,
    capacity_mb: float,
    ttl_s: float,
) -> int:
    if footprint_mb > capacity_mb:
        return 2
    for slot in range(service_slots.size):
        if service_slots[slot] == service and start_s - last_completion[slot] <= ttl_s:
            return 0
    return 1


@nb.njit(cache=True)
def _cache_commit(
    service: int,
    footprint_mb: float,
    start_s: float,
    completion_s: float,
    service_slots: np.ndarray,
    last_completion: np.ndarray,
    memory_slots: np.ndarray,
    capacity_mb: float,
    ttl_s: float,
) -> tuple[int, int]:
    occupied = 0.0
    empty = -1
    hit = -1
    evictions = 0
    for slot in range(service_slots.size):
        sid = service_slots[slot]
        if sid >= 0 and start_s - last_completion[slot] > ttl_s:
            service_slots[slot] = -1
            memory_slots[slot] = 0.0
            sid = -1
        if sid < 0:
            if empty < 0:
                empty = slot
        else:
            occupied += memory_slots[slot]
            if sid == service:
                hit = slot
    if hit >= 0:
        last_completion[hit] = completion_s
        return 0, 0
    if footprint_mb > capacity_mb:
        return 2, 0
    while occupied + footprint_mb > capacity_mb:
        victim = -1
        oldest = np.inf
        for slot in range(service_slots.size):
            if service_slots[slot] >= 0 and last_completion[slot] < oldest:
                victim = slot
                oldest = last_completion[slot]
        if victim < 0:
            return 2, evictions
        occupied -= memory_slots[victim]
        service_slots[victim] = -1
        memory_slots[victim] = 0.0
        empty = victim
        evictions += 1
    if empty < 0:
        # A cardinality sensitivity can bind before the byte capacity.  Evict
        # the same least-recently-used entry rather than treating a cold access
        # as infeasible.
        oldest = np.inf
        for slot in range(service_slots.size):
            if service_slots[slot] >= 0 and last_completion[slot] < oldest:
                empty = slot
                oldest = last_completion[slot]
        if empty >= 0:
            occupied -= memory_slots[empty]
            service_slots[empty] = -1
            memory_slots[empty] = 0.0
            evictions += 1
    if empty < 0:
        return 2, evictions
    service_slots[empty] = service
    memory_slots[empty] = footprint_mb
    last_completion[empty] = completion_s
    return 1, evictions


@nb.njit(cache=True)
def _thermal_phase(
    temperature_c: float,
    duration_s: float,
    power_w: float,
    idle_w: float,
    idle_temperature_c: float,
    gain_c_per_w: float,
    tau_s: float,
) -> float:
    steady = idle_temperature_c + gain_c_per_w * (power_w - idle_w)
    return steady + (temperature_c - steady) * math.exp(-duration_s / tau_s)


@nb.njit(cache=True)
def _thermal_segment_peak(
    temperature_c: float,
    segment_start_s: float,
    segment_stop_s: float,
    power_w: float,
    idle_w: float,
    idle_temperature_c: float,
    gain_c_per_w: float,
    tau_s: float,
    interval_start_s: float,
    interval_stop_s: float,
) -> float:
    """Exact peak of one constant-power thermal phase inside a time interval.

    A first-order phase is monotone toward its equilibrium, so its maximum on
    an overlap is attained at one of the two clipped endpoints.
    """

    lower = max(segment_start_s, interval_start_s)
    upper = min(segment_stop_s, interval_stop_s)
    if upper <= lower:
        return -np.inf
    at_lower = _thermal_phase(
        temperature_c,
        lower - segment_start_s,
        power_w,
        idle_w,
        idle_temperature_c,
        gain_c_per_w,
        tau_s,
    )
    at_upper = _thermal_phase(
        temperature_c,
        upper - segment_start_s,
        power_w,
        idle_w,
        idle_temperature_c,
        gain_c_per_w,
        tau_s,
    )
    return max(at_lower, at_upper)


@nb.njit(cache=True)
def _thermal_job(
    temperature_c: float,
    state_time_s: float,
    start_s: float,
    startup_s: float,
    service_s: float,
    idle_w: float,
    active_w: float,
    startup_incremental_power_fraction: float,
    idle_temperature_c: float,
    gain_c_per_w: float,
    tau_s: float,
) -> tuple[float, float, float]:
    idle_gap = max(0.0, start_s - state_time_s)
    at_start = _thermal_phase(
        temperature_c,
        idle_gap,
        idle_w,
        idle_w,
        idle_temperature_c,
        gain_c_per_w,
        tau_s,
    )
    startup_power = idle_w + startup_incremental_power_fraction * (active_w - idle_w)
    at_service = _thermal_phase(
        at_start,
        startup_s,
        startup_power,
        idle_w,
        idle_temperature_c,
        gain_c_per_w,
        tau_s,
    )
    at_finish = _thermal_phase(
        at_service,
        service_s,
        active_w,
        idle_w,
        idle_temperature_c,
        gain_c_per_w,
        tau_s,
    )
    return at_start, at_service, at_finish


@nb.njit(cache=True)
def _interval_overlap(start_s: float, stop_s: float, lower_s: float, upper_s: float) -> float:
    """Return exact overlap with a half-open wall-clock measurement interval."""

    return max(0.0, min(stop_s, upper_s) - max(start_s, lower_s))


@nb.njit(cache=True)
def _route_preview(
    node: int,
    release_s: float,
    payload_bytes: float,
    link_available: np.ndarray,
    route_links: np.ndarray,
    route_length: np.ndarray,
    link_rate_bps: np.ndarray,
    link_fixed_s: np.ndarray,
    link_propagation_s: np.ndarray,
    background: np.ndarray,
    z_by_hop: np.ndarray,
    stochastic: int,
    residual_exponent: float,
    minimum_capacity_fraction: float,
) -> tuple[float, float, float]:
    first_arrival = release_s
    last_completion = release_s
    queue_total = 0.0
    jitter_total = 0.0
    for hop in range(route_length[node]):
        link = route_links[node, hop]
        start = max(first_arrival, link_available[link])
        service_minute = min(max(int(start // 60.0), 0), background.shape[0] - 1)
        utilization = min(max(float(background[service_minute, link]), 0.0), 0.95)
        capacity_fraction = max(
            (1.0 - utilization) ** residual_exponent,
            minimum_capacity_fraction,
        )
        effective_rate = link_rate_bps[link] * capacity_fraction
        serialization = payload_bytes * 8.0 / effective_rate
        service = serialization + link_fixed_s[link]
        queue_total += start - first_arrival
        link_finish = start + service
        jitter_mean = hop_jitter_mean(serialization, utilization)
        if stochastic:
            probability = 0.5 * (
                1.0 + math.erf(z_by_hop[hop] / math.sqrt(2.0))
            )
            probability = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
            jitter = -jitter_mean * math.log1p(-probability)
        else:
            jitter = jitter_mean
        jitter_total += jitter
        link_busy_finish = link_finish + jitter
        first_arrival = link_busy_finish + link_propagation_s[link]
        last_completion = link_busy_finish + link_propagation_s[link]
    return last_completion, queue_total, jitter_total


@nb.njit(cache=True)
def _route_commit(
    node: int,
    release_s: float,
    payload_bytes: float,
    link_available: np.ndarray,
    route_links: np.ndarray,
    route_length: np.ndarray,
    link_rate_bps: np.ndarray,
    link_fixed_s: np.ndarray,
    link_propagation_s: np.ndarray,
    background: np.ndarray,
    z_by_hop: np.ndarray,
    stochastic: int,
    residual_exponent: float,
    minimum_capacity_fraction: float,
) -> tuple[float, float, float]:
    first_arrival = release_s
    last_completion = release_s
    queue_total = 0.0
    jitter_total = 0.0
    for hop in range(route_length[node]):
        link = route_links[node, hop]
        start = max(first_arrival, link_available[link])
        service_minute = min(max(int(start // 60.0), 0), background.shape[0] - 1)
        utilization = min(max(float(background[service_minute, link]), 0.0), 0.95)
        capacity_fraction = max(
            (1.0 - utilization) ** residual_exponent,
            minimum_capacity_fraction,
        )
        effective_rate = link_rate_bps[link] * capacity_fraction
        serialization = payload_bytes * 8.0 / effective_rate
        service = serialization + link_fixed_s[link]
        queue_total += start - first_arrival
        link_finish = start + service
        jitter_mean = hop_jitter_mean(serialization, utilization)
        if stochastic:
            probability = 0.5 * (
                1.0 + math.erf(z_by_hop[hop] / math.sqrt(2.0))
            )
            probability = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
            jitter = -jitter_mean * math.log1p(-probability)
        else:
            jitter = jitter_mean
        jitter_total += jitter
        link_busy_finish = link_finish + jitter
        link_available[link] = link_busy_finish
        first_arrival = link_busy_finish + link_propagation_s[link]
        last_completion = link_busy_finish + link_propagation_s[link]
    return last_completion, queue_total, jitter_total


@nb.njit(cache=True)
def _advance_bucket(tokens: float, state_s: float, now_s: float, rate: float, capacity: float) -> tuple[float, float]:
    if now_s > state_s:
        tokens = min(capacity, tokens + (now_s - state_s) * rate)
        state_s = now_s
    return tokens, state_s


@nb.njit(cache=True)
def _advance_debt(
    debt_j: float, state_s: float, now_s: float, credit_rate_j_per_s: float
) -> tuple[float, float]:
    if now_s > state_s:
        debt_j = max(0.0, debt_j - (now_s - state_s) * credit_rate_j_per_s)
        state_s = now_s
    return debt_j, state_s


@nb.njit(cache=True)
def _request_deadline(
    request: int,
    release_s: np.ndarray,
    function_id: np.ndarray,
    tenant_by_function: np.ndarray,
    p99_ms: np.ndarray,
    minimum_deadline_s: np.ndarray,
    tail_multiplier: float,
) -> float:
    function = function_id[request]
    day = min(int(release_s[request] // 86400.0), p99_ms.shape[1] - 1)
    tenant = tenant_by_function[function]
    return max(minimum_deadline_s[tenant], tail_multiplier * p99_ms[function, day] / 1000.0)


@nb.njit(cache=True)
def _empirical_duration_ms(
    values: np.ndarray,
    probabilities: np.ndarray,
    standard_normal: float,
    variability_multiplier: float,
) -> float:
    probability = 0.5 * (
        1.0 + math.erf(standard_normal * variability_multiplier / math.sqrt(2.0))
    )
    if probability <= probabilities[0]:
        return values[0]
    for index in range(1, len(probabilities)):
        if probability <= probabilities[index]:
            width = probabilities[index] - probabilities[index - 1]
            fraction = (probability - probabilities[index - 1]) / width
            return values[index - 1] + fraction * (values[index] - values[index - 1])
    return values[-1]


@nb.njit(cache=True)
def _piecewise_quantile(
    values: np.ndarray, probabilities: np.ndarray, probability: float
) -> float:
    if probability <= probabilities[0]:
        return values[0]
    for index in range(1, len(probabilities)):
        if probability <= probabilities[index]:
            fraction = (probability - probabilities[index - 1]) / (
                probabilities[index] - probabilities[index - 1]
            )
            return values[index - 1] + fraction * (values[index] - values[index - 1])
    return values[-1]


@nb.njit(cache=True)
def _simulate_kernel(
    method: int,
    release_s: np.ndarray,
    evaluation_start_s: float,
    evaluation_end_s: float,
    function_id: np.ndarray,
    service_z: np.ndarray,
    network_z: np.ndarray,
    startup_z: np.ndarray,
    p50_ms: np.ndarray,
    p99_ms: np.ndarray,
    duration_quantiles_ms: np.ndarray,
    duration_quantile_probabilities: np.ndarray,
    memory_mb: np.ndarray,
    cache_identity_by_function: np.ndarray,
    cache_footprint_mb: np.ndarray,
    tenant_by_function: np.ndarray,
    requirement_mask: np.ndarray,
    capability_mask: np.ndarray,
    usable_memory_mb: np.ndarray,
    service_rate: np.ndarray,
    idle_w: np.ndarray,
    incremental_w: np.ndarray,
    energy_budget_rate_base: np.ndarray,
    tenant_energy_share: np.ndarray,
    idle_temperature_c: np.ndarray,
    thermal_gain: np.ndarray,
    thermal_tau: np.ndarray,
    startup_mu: np.ndarray,
    startup_sigma: np.ndarray,
    route_links: np.ndarray,
    route_length: np.ndarray,
    link_rate_bps: np.ndarray,
    link_fixed_s: np.ndarray,
    link_propagation_s: np.ndarray,
    background_pred: np.ndarray,
    background_actual: np.ndarray,
    service_pressure: np.ndarray,
    stratum_by_function: np.ndarray,
    capacity_descriptor: np.ndarray,
    predicted_power_factor: float,
    actual_power_factor: np.ndarray,
    network_residual_exponent: float,
    network_minimum_capacity_fraction: float,
    scd_v_joules: float,
    minimum_deadline_s: np.ndarray,
    deadline_tail_multiplier: float,
    service_prediction_quantile: float,
    startup_prediction_z: float,
    cache_ttl_s: float,
    cache_footprint_multiplier: float,
    cache_slots: int,
    payload_multiplier: float,
    payload_minimum_bytes: float,
    payload_sqrt_memory_coefficient_bytes: float,
    startup_incremental_power_fraction: float,
    service_variability_multiplier: float,
    startup_variability_multiplier: float,
    energy_budget_factor: float,
    energy_burst_s: float,
    energy_charge_margin: float,
    idle_allocation_fraction: float,
    request_cap_per_tenant: int,
    request_window_s: float,
    admission_guard: int,
    thermal_limit_c: float,
    thermal_guard: int,
    energy_guard: int,
    deadline_guard: int,
    deadline_guard_multiplier: float,
    thermal_derate_start_c: float,
    thermal_derate_slope_per_c: float,
    thermal_minimum_service_multiplier: float,
) -> tuple:
    requests = len(release_s)
    nodes = len(service_rate)
    completion_s = np.full(requests, np.nan, dtype=np.float64)
    selected_node = np.full(requests, -1, dtype=np.int16)
    rejection_code = np.zeros(requests, dtype=np.uint8)
    predicted_cold = np.zeros(requests, dtype=np.uint8)
    actual_cold = np.zeros(requests, dtype=np.uint8)
    queue_s = np.zeros(requests, dtype=np.float32)
    jitter_s = np.zeros(requests, dtype=np.float32)
    task_energy_j = np.zeros(requests, dtype=np.float32)
    activity_energy_j = np.zeros(requests, dtype=np.float32)
    charged_energy_j = np.zeros(requests, dtype=np.float32)
    policy_predicted_completion_s = np.full(requests, np.nan, dtype=np.float64)
    policy_shift_at_dispatch_s = np.zeros(requests, dtype=np.float64)

    pred_device_available = np.zeros(nodes, dtype=np.float64)
    policy_queue_shift_s = np.zeros(nodes, dtype=np.float64)
    actual_device_available = np.zeros(nodes, dtype=np.float64)
    pred_link_available = np.zeros(len(link_rate_bps), dtype=np.float64)
    no_link_queue_state = np.zeros(len(link_rate_bps), dtype=np.float64)
    actual_link_available = np.zeros(len(link_rate_bps), dtype=np.float64)
    pred_temperature = idle_temperature_c.copy()
    actual_temperature = idle_temperature_c.copy()
    pred_temperature_time = np.zeros(nodes, dtype=np.float64)
    actual_temperature_time = np.zeros(nodes, dtype=np.float64)

    pred_cache_service = np.full((nodes, cache_slots), -1, dtype=np.int32)
    pred_cache_last = np.zeros((nodes, cache_slots), dtype=np.float64)
    pred_cache_memory = np.zeros((nodes, cache_slots), dtype=np.float32)
    actual_cache_service = np.full((nodes, cache_slots), -1, dtype=np.int32)
    actual_cache_last = np.zeros((nodes, cache_slots), dtype=np.float64)
    actual_cache_memory = np.zeros((nodes, cache_slots), dtype=np.float32)

    budget_rate = energy_budget_rate_base * energy_budget_factor
    energy_capacity = budget_rate * energy_burst_s
    energy_tokens = energy_capacity.copy()
    energy_token_time = np.zeros(nodes, dtype=np.float64)
    energy_debt = np.zeros(nodes, dtype=np.float64)
    energy_debt_time = np.zeros(nodes, dtype=np.float64)
    tenant_energy_capacity = energy_capacity.reshape(nodes, 1) * tenant_energy_share.reshape(1, 4)
    tenant_energy_tokens = tenant_energy_capacity.copy()
    tenant_energy_token_time = np.zeros((nodes, 4), dtype=np.float64)
    tenant_energy_debt = np.zeros((nodes, 4), dtype=np.float64)
    tenant_energy_debt_time = np.zeros((nodes, 4), dtype=np.float64)
    max_energy_debt = 0.0
    max_tenant_energy_debt = 0.0
    request_tokens = np.full(4, request_cap_per_tenant, dtype=np.int64)
    request_window = np.full(4, -1, dtype=np.int64)
    window_commitment_j = np.zeros(nodes, dtype=np.float64)
    commitment_window = -1

    completion_next = np.full(requests, -1, dtype=np.int64)
    completion_head = np.full(nodes, -1, dtype=np.int64)
    completion_tail = np.full(nodes, -1, dtype=np.int64)
    node_assignments = np.zeros(nodes, dtype=np.int64)
    tenant_offered = np.zeros(4, dtype=np.int64)
    tenant_admitted = np.zeros(4, dtype=np.int64)
    tenant_success = np.zeros(4, dtype=np.int64)
    rr_cursor = 0
    cache_evictions_pred = 0
    cache_evictions_actual = 0
    false_warm = 0
    false_cold = 0
    actual_cold_count = 0
    predicted_cold_count = 0
    peak_temperature = float(np.max(actual_temperature))
    thermal_violations = 0
    thermal_derate_count = 0
    energy_underestimation_count = 0
    energy_shortfall = 0.0
    evaluation_activity_energy = 0.0

    candidate_score = np.empty(nodes, dtype=np.float64)
    candidate_pred_start = np.empty(nodes, dtype=np.float64)
    candidate_pred_duration = np.empty(nodes, dtype=np.float64)
    candidate_pred_startup = np.empty(nodes, dtype=np.float64)
    candidate_pred_service = np.empty(nodes, dtype=np.float64)
    candidate_pred_energy = np.empty(nodes, dtype=np.float64)
    candidate_pred_cold = np.empty(nodes, dtype=np.uint8)
    candidate_on_time = np.zeros(nodes, dtype=np.uint8)
    candidate_compute_wait = np.empty(nodes, dtype=np.float64)
    candidate_network = np.empty(nodes, dtype=np.float64)
    candidate_switch_wait = np.empty(nodes, dtype=np.float64)
    candidate_temperature = np.empty(nodes, dtype=np.float64)
    candidate_current_ratio = np.empty(nodes, dtype=np.float64)
    candidate_projected_ratio = np.empty(nodes, dtype=np.float64)
    candidate_debt_ratio = np.empty(nodes, dtype=np.float64)
    candidate_tenant_ratio = np.empty(nodes, dtype=np.float64)

    for request in range(requests):
        now = release_s[request]
        in_evaluation = evaluation_start_s <= now < evaluation_end_s
        # Process all outcomes that are observable by this release.  This is a
        # global chronological merge of the per-device FIFO completion lists.
        while True:
            completion_node = -1
            completion_request = -1
            completion_time = now + 1.0
            for node in range(nodes):
                head = completion_head[node]
                if head >= 0 and completion_s[head] <= now and completion_s[head] < completion_time:
                    completion_node = node
                    completion_request = head
                    completion_time = completion_s[head]
            if completion_node < 0:
                break
            completion_head[completion_node] = completion_next[completion_request]
            if completion_head[completion_node] < 0:
                completion_tail[completion_node] = -1
            expected_completion = (
                policy_predicted_completion_s[completion_request]
                + policy_queue_shift_s[completion_node]
                - policy_shift_at_dispatch_s[completion_request]
            )
            prediction_error = completion_time - expected_completion
            policy_queue_shift_s[completion_node] += prediction_error
            pred_device_available[completion_node] = max(
                completion_time,
                pred_device_available[completion_node] + prediction_error,
            )
            for slot in range(cache_slots):
                if (
                    pred_cache_service[completion_node, slot] >= 0
                    and pred_cache_last[completion_node, slot] >= expected_completion - 1.0e-12
                ):
                    pred_cache_last[completion_node, slot] += prediction_error
            pred_temperature_time[completion_node] = max(
                completion_time,
                pred_temperature_time[completion_node] + prediction_error,
            )
            if completion_head[completion_node] < 0:
                pred_device_available[completion_node] = completion_time
                pred_temperature[completion_node] = actual_temperature[completion_node]
                pred_temperature_time[completion_node] = completion_time
                for slot in range(cache_slots):
                    pred_cache_service[completion_node, slot] = actual_cache_service[
                        completion_node, slot
                    ]
                    pred_cache_last[completion_node, slot] = actual_cache_last[
                        completion_node, slot
                    ]
                    pred_cache_memory[completion_node, slot] = actual_cache_memory[
                        completion_node, slot
                    ]
            tokens, token_time = _advance_bucket(
                energy_tokens[completion_node],
                energy_token_time[completion_node],
                completion_time,
                budget_rate[completion_node],
                energy_capacity[completion_node],
            )
            energy_tokens[completion_node] = tokens
            energy_token_time[completion_node] = token_time
            energy_debt[completion_node], energy_debt_time[completion_node] = _advance_debt(
                energy_debt[completion_node],
                energy_debt_time[completion_node],
                completion_time,
                budget_rate[completion_node],
            )
            tenant_done = tenant_by_function[function_id[completion_request]]
            if method != 13:
                tenant_tokens, tenant_token_time = _advance_bucket(
                    tenant_energy_tokens[completion_node, tenant_done],
                    tenant_energy_token_time[completion_node, tenant_done],
                    completion_time,
                    budget_rate[completion_node] * tenant_energy_share[tenant_done],
                    tenant_energy_capacity[completion_node, tenant_done],
                )
                tenant_energy_tokens[completion_node, tenant_done] = tenant_tokens
                tenant_energy_token_time[completion_node, tenant_done] = tenant_token_time
                (
                    tenant_energy_debt[completion_node, tenant_done],
                    tenant_energy_debt_time[completion_node, tenant_done],
                ) = _advance_debt(
                    tenant_energy_debt[completion_node, tenant_done],
                    tenant_energy_debt_time[completion_node, tenant_done],
                    completion_time,
                    budget_rate[completion_node] * tenant_energy_share[tenant_done],
                )
            deadline_done = _request_deadline(
                completion_request,
                release_s,
                function_id,
                tenant_by_function,
                p99_ms,
                minimum_deadline_s,
                deadline_tail_multiplier,
            )
            completed_in_evaluation = (
                evaluation_start_s
                <= release_s[completion_request]
                < evaluation_end_s
            )
            if completion_time - release_s[completion_request] <= deadline_done:
                if completed_in_evaluation:
                    tenant_success[tenant_done] += 1

        function = function_id[request]
        tenant = tenant_by_function[function]
        if in_evaluation:
            tenant_offered[tenant] += 1
        current_request_window = int(math.floor(now / request_window_s))
        if commitment_window != current_request_window:
            window_commitment_j[:] = 0.0
            commitment_window = current_request_window
        # Link completion is observable only once the realized FCFS resource
        # has become idle.  Resetting the reservation then is causal and avoids
        # retaining prediction error indefinitely.  A future realized
        # departure is never copied into policy state.
        if method != 14:
            for link in range(len(pred_link_available)):
                if actual_link_available[link] <= now:
                    pred_link_available[link] = now
        if admission_guard:
            if request_window[tenant] != current_request_window:
                request_window[tenant] = current_request_window
                request_tokens[tenant] = request_cap_per_tenant
            if request_tokens[tenant] < 1:
                rejection_code[request] = 1
                continue
            request_tokens[tenant] -= 1

        for node in range(nodes):
            energy_tokens[node], energy_token_time[node] = _advance_bucket(
                energy_tokens[node],
                energy_token_time[node],
                now,
                budget_rate[node],
                energy_capacity[node],
            )
            energy_debt[node], energy_debt_time[node] = _advance_debt(
                energy_debt[node], energy_debt_time[node], now, budget_rate[node]
            )
            for tenant_index in range(4):
                (
                    tenant_energy_tokens[node, tenant_index],
                    tenant_energy_token_time[node, tenant_index],
                ) = _advance_bucket(
                    tenant_energy_tokens[node, tenant_index],
                    tenant_energy_token_time[node, tenant_index],
                    now,
                    budget_rate[node] * tenant_energy_share[tenant_index],
                    tenant_energy_capacity[node, tenant_index],
                )
                if method != 13:
                    (
                        tenant_energy_debt[node, tenant_index],
                        tenant_energy_debt_time[node, tenant_index],
                    ) = _advance_debt(
                        tenant_energy_debt[node, tenant_index],
                        tenant_energy_debt_time[node, tenant_index],
                        now,
                        budget_rate[node] * tenant_energy_share[tenant_index],
                    )

        day = min(int(now // 86400.0), p50_ms.shape[1] - 1)
        predicted_reference_service_s = max(
            _piecewise_quantile(
                duration_quantiles_ms[function, day],
                duration_quantile_probabilities,
                service_prediction_quantile,
            )
            / 1000.0,
            1.0e-6,
        )
        cache_service = cache_identity_by_function[function]
        footprint_mb = cache_footprint_mb[function] * cache_footprint_multiplier
        payload_bytes = (
            max(
                payload_minimum_bytes,
                math.sqrt(memory_mb[function])
                * payload_sqrt_memory_coefficient_bytes,
            )
            * payload_multiplier
        )
        deadline_s = max(
            minimum_deadline_s[tenant],
            deadline_tail_multiplier * p99_ms[function, day] / 1000.0,
        )
        minute = min(int(now // 60.0), background_pred.shape[0] - 1)
        compatible_seen = 0
        energy_blocked = 0
        thermal_blocked = 0
        deadline_blocked = 0
        minimum_compatible_energy = np.inf

        for node in range(nodes):
            candidate_score[node] = np.inf
            candidate_on_time[node] = 0
            if (capability_mask[node] & requirement_mask[function]) != requirement_mask[function]:
                continue
            if footprint_mb > usable_memory_mb[node]:
                continue
            compatible_seen += 1
            net_finish, net_queue, _ = _route_preview(
                node,
                now,
                payload_bytes,
                no_link_queue_state if method == 14 else pred_link_available,
                route_links,
                route_length,
                link_rate_bps,
                link_fixed_s,
                link_propagation_s,
                background_pred,
                network_z[request],
                0,
                network_residual_exponent,
                network_minimum_capacity_fraction,
            )
            start_s = max(net_finish, pred_device_available[node])
            status = _cache_status(
                cache_service,
                footprint_mb,
                start_s,
                pred_cache_service[node],
                pred_cache_last[node],
                usable_memory_mb[node],
                cache_ttl_s,
            )
            if status == 2:
                continue
            cold = 1 if status == 1 else 0
            startup_prediction = math.exp(
                startup_mu[node] + startup_sigma[node] * startup_prediction_z
            )
            predicted_startup = cold * startup_prediction
            predicted_active_w = (
                idle_w[node] + incremental_w[node] * predicted_power_factor
            )
            temperature_at_start = _thermal_phase(
                pred_temperature[node],
                max(0.0, start_s - pred_temperature_time[node]),
                idle_w[node],
                idle_w[node],
                idle_temperature_c[node],
                thermal_gain[node],
                thermal_tau[node],
            )
            temperature_at_service = _thermal_phase(
                temperature_at_start,
                predicted_startup,
                idle_w[node]
                + startup_incremental_power_fraction
                * incremental_w[node]
                * predicted_power_factor,
                idle_w[node],
                idle_temperature_c[node],
                thermal_gain[node],
                thermal_tau[node],
            )
            predicted_derate = max(
                thermal_minimum_service_multiplier,
                1.0
                - thermal_derate_slope_per_c
                * max(0.0, temperature_at_service - thermal_derate_start_c),
            )
            predicted_service = predicted_reference_service_s / (
                service_rate[node] * predicted_derate
            )
            duration = predicted_startup + predicted_service
            _, _, temperature_finish = _thermal_job(
                pred_temperature[node],
                pred_temperature_time[node],
                start_s,
                predicted_startup,
                predicted_service,
                idle_w[node],
                predicted_active_w,
                startup_incremental_power_fraction,
                idle_temperature_c[node],
                thermal_gain[node],
                thermal_tau[node],
            )
            pred_completion = start_s + duration
            pred_latency = pred_completion - now
            nominal_pred_energy = (
                incremental_w[node]
                * (predicted_service + startup_incremental_power_fraction * predicted_startup)
                + idle_allocation_fraction
                * idle_w[node]
                * (pred_completion - now)
            )
            charge = nominal_pred_energy * predicted_power_factor * energy_charge_margin
            minimum_compatible_energy = min(minimum_compatible_energy, charge)
            if thermal_guard and max(
                temperature_at_start,
                temperature_at_service,
                temperature_finish,
            ) > thermal_limit_c:
                thermal_blocked += 1
                continue
            if deadline_guard and pred_latency > deadline_guard_multiplier * deadline_s:
                deadline_blocked += 1
                continue
            if energy_guard and (
                charge > energy_tokens[node] + 1.0e-12
                or (
                    method != 13
                    and charge > tenant_energy_tokens[node, tenant] + 1.0e-12
                )
            ):
                energy_blocked += 1
                continue
            one_window_budget = max(budget_rate[node] * request_window_s, 1.0e-9)
            candidate_score[node] = 0.0
            candidate_pred_start[node] = start_s
            candidate_pred_duration[node] = duration
            candidate_pred_startup[node] = predicted_startup
            candidate_pred_service[node] = predicted_service
            candidate_pred_energy[node] = charge
            candidate_pred_cold[node] = cold
            candidate_on_time[node] = 1 if pred_latency <= deadline_s else 0
            candidate_compute_wait[node] = max(0.0, pred_device_available[node] - now)
            candidate_network[node] = net_finish - now
            candidate_switch_wait[node] = net_queue
            candidate_temperature[node] = temperature_finish
            candidate_current_ratio[node] = window_commitment_j[node] / one_window_budget
            candidate_projected_ratio[node] = (
                window_commitment_j[node] + charge
            ) / one_window_budget
            candidate_debt_ratio[node] = energy_debt[node] / one_window_budget
            candidate_tenant_ratio[node] = tenant_energy_debt[node, tenant] / max(
                one_window_budget * tenant_energy_share[tenant], 1.0e-9
            )

        request_pressure = service_pressure[minute, stratum_by_function[function]]
        for node in range(nodes):
            if not math.isfinite(candidate_score[node]):
                continue
            pred_latency = candidate_pred_start[node] + candidate_pred_duration[node] - now
            core_method = 10 if method == 14 else method
            h = operating_penalty(
                pred_latency,
                deadline_s,
                candidate_pred_energy[node],
                minimum_compatible_energy,
                candidate_compute_wait[node],
                candidate_network[node],
                candidate_pred_service[node],
                candidate_switch_wait[node],
                request_pressure,
                capacity_descriptor[node],
                candidate_current_ratio[node],
                candidate_projected_ratio[node],
                candidate_debt_ratio[node],
                candidate_temperature[node],
                thermal_limit_c,
                core_method,
            )
            if method == 1:
                score = candidate_pred_start[node] + candidate_pred_duration[node]
            elif method == 2:
                score = candidate_pred_energy[node]
            else:
                score = scd_candidate_score(
                    h,
                    candidate_pred_energy[node],
                    candidate_debt_ratio[node],
                    candidate_tenant_ratio[node],
                    scd_v_joules,
                    core_method,
                )
            candidate_score[node] = score

        chosen = -1
        if method == 4:
            for offset in range(nodes):
                node = (rr_cursor + offset) % nodes
                if math.isfinite(candidate_score[node]):
                    chosen = node
                    rr_cursor = (node + 1) % nodes
                    break
        else:
            best = np.inf
            require_on_time = 0
            if method in (6, 7, 8, 9, 10, 11, 13, 14):
                for node in range(nodes):
                    if math.isfinite(candidate_score[node]) and candidate_on_time[node] == 1:
                        require_on_time = 1
                        break
            for node in range(nodes):
                if not math.isfinite(candidate_score[node]):
                    continue
                if require_on_time and candidate_on_time[node] == 0:
                    continue
                if candidate_score[node] < best:
                    best = candidate_score[node]
                    chosen = node
        if chosen < 0:
            if compatible_seen == 0:
                rejection_code[request] = 3
            elif energy_blocked > 0:
                rejection_code[request] = 2
            elif thermal_blocked > 0:
                rejection_code[request] = 4
            elif deadline_blocked > 0:
                rejection_code[request] = 5
            else:
                rejection_code[request] = 3
            continue

        if method == 14:
            pred_net_finish, _, _ = _route_preview(
                chosen,
                now,
                payload_bytes,
                no_link_queue_state,
                route_links,
                route_length,
                link_rate_bps,
                link_fixed_s,
                link_propagation_s,
                background_pred,
                network_z[request],
                0,
                network_residual_exponent,
                network_minimum_capacity_fraction,
            )
        else:
            pred_net_finish, _, _ = _route_commit(
                chosen,
                now,
                payload_bytes,
                pred_link_available,
                route_links,
                route_length,
                link_rate_bps,
                link_fixed_s,
                link_propagation_s,
                background_pred,
                network_z[request],
                0,
                network_residual_exponent,
                network_minimum_capacity_fraction,
            )
        pred_start = max(pred_net_finish, pred_device_available[chosen])
        pred_duration = candidate_pred_duration[chosen]
        pred_completion = pred_start + pred_duration
        pred_status, evicted = _cache_commit(
            cache_service,
            footprint_mb,
            pred_start,
            pred_completion,
            pred_cache_service[chosen],
            pred_cache_last[chosen],
            pred_cache_memory[chosen],
            usable_memory_mb[chosen],
            cache_ttl_s,
        )
        if in_evaluation:
            cache_evictions_pred += evicted
        if pred_status == 2:
            raise RuntimeError("predicted cache commit contradicted feasible preview")
        predicted_cold[request] = 1 if pred_status == 1 else 0
        if in_evaluation:
            predicted_cold_count += predicted_cold[request]
        _, _, pred_finish_temperature = _thermal_job(
            pred_temperature[chosen],
            pred_temperature_time[chosen],
            pred_start,
            candidate_pred_startup[chosen],
            candidate_pred_service[chosen],
            idle_w[chosen],
            idle_w[chosen] + incremental_w[chosen] * predicted_power_factor,
            startup_incremental_power_fraction,
            idle_temperature_c[chosen],
            thermal_gain[chosen],
            thermal_tau[chosen],
        )
        pred_temperature[chosen] = pred_finish_temperature
        pred_temperature_time[chosen] = pred_completion
        pred_device_available[chosen] = pred_completion

        actual_net_finish, actual_queue, actual_jitter = _route_commit(
            chosen,
            now,
            payload_bytes,
            actual_link_available,
            route_links,
            route_length,
            link_rate_bps,
            link_fixed_s,
            link_propagation_s,
            background_actual,
            network_z[request],
            1,
            network_residual_exponent,
            network_minimum_capacity_fraction,
        )
        actual_start = max(actual_net_finish, actual_device_available[chosen])
        actual_status = _cache_status(
            cache_service,
            footprint_mb,
            actual_start,
            actual_cache_service[chosen],
            actual_cache_last[chosen],
            usable_memory_mb[chosen],
            cache_ttl_s,
        )
        actual_is_cold = 1 if actual_status == 1 else 0
        actual_service_reference = max(
            _empirical_duration_ms(
                duration_quantiles_ms[function, day],
                duration_quantile_probabilities,
                service_z[request],
                service_variability_multiplier,
            )
            / 1000.0,
            1.0e-6,
        )
        actual_startup = 0.0
        if actual_is_cold:
            actual_startup = math.exp(
                startup_mu[chosen]
                + startup_sigma[chosen]
                * startup_variability_multiplier
                * startup_z[request]
            )
        previous_actual_temperature = actual_temperature[chosen]
        previous_actual_temperature_time = actual_temperature_time[chosen]
        actual_temperature_at_start = _thermal_phase(
            previous_actual_temperature,
            max(0.0, actual_start - previous_actual_temperature_time),
            idle_w[chosen],
            idle_w[chosen],
            idle_temperature_c[chosen],
            thermal_gain[chosen],
            thermal_tau[chosen],
        )
        power_minute = min(
            max(int(actual_start // 60.0), 0), actual_power_factor.shape[0] - 1
        )
        realized_power_factor = actual_power_factor[power_minute, chosen]
        realized_active_w = idle_w[chosen] + incremental_w[chosen] * realized_power_factor
        actual_temperature_at_service = _thermal_phase(
            actual_temperature_at_start,
            actual_startup,
            idle_w[chosen]
            + startup_incremental_power_fraction
            * incremental_w[chosen]
            * realized_power_factor,
            idle_w[chosen],
            idle_temperature_c[chosen],
            thermal_gain[chosen],
            thermal_tau[chosen],
        )
        actual_derate = max(
            thermal_minimum_service_multiplier,
            1.0
            - thermal_derate_slope_per_c
            * max(0.0, actual_temperature_at_service - thermal_derate_start_c),
        )
        actual_service = actual_service_reference / (
            service_rate[chosen] * actual_derate
        )
        actual_duration = actual_service + actual_startup
        actual_completion = actual_start + actual_duration
        startup_end = actual_start + actual_startup
        evaluation_activity_energy += (
            startup_incremental_power_fraction
            * incremental_w[chosen]
            * realized_power_factor
            * _interval_overlap(
                actual_start,
                startup_end,
                evaluation_start_s,
                evaluation_end_s,
            )
            + incremental_w[chosen]
            * realized_power_factor
            * _interval_overlap(
                startup_end,
                actual_completion,
                evaluation_start_s,
                evaluation_end_s,
            )
        )
        committed_status, evicted_actual = _cache_commit(
            cache_service,
            footprint_mb,
            actual_start,
            actual_completion,
            actual_cache_service[chosen],
            actual_cache_last[chosen],
            actual_cache_memory[chosen],
            usable_memory_mb[chosen],
            cache_ttl_s,
        )
        if in_evaluation:
            cache_evictions_actual += evicted_actual
        if committed_status == 2:
            raise RuntimeError("realized cache commit contradicted feasible preview")
        actual_cold[request] = actual_is_cold
        if in_evaluation:
            actual_cold_count += actual_is_cold
            if predicted_cold[request] == 0 and actual_is_cold == 1:
                false_warm += 1
            elif predicted_cold[request] == 1 and actual_is_cold == 0:
                false_cold += 1
        _, _, actual_finish_temperature = _thermal_job(
            actual_temperature[chosen],
            actual_temperature_time[chosen],
            actual_start,
            actual_startup,
            actual_service,
            idle_w[chosen],
            realized_active_w,
            startup_incremental_power_fraction,
            idle_temperature_c[chosen],
            thermal_gain[chosen],
            thermal_tau[chosen],
        )
        actual_temperature[chosen] = actual_finish_temperature
        actual_temperature_time[chosen] = actual_completion
        actual_device_available[chosen] = actual_completion
        peak_temperature = max(
            peak_temperature,
            _thermal_segment_peak(
                previous_actual_temperature,
                previous_actual_temperature_time,
                actual_start,
                idle_w[chosen],
                idle_w[chosen],
                idle_temperature_c[chosen],
                thermal_gain[chosen],
                thermal_tau[chosen],
                evaluation_start_s,
                evaluation_end_s,
            ),
            _thermal_segment_peak(
                actual_temperature_at_start,
                actual_start,
                startup_end,
                idle_w[chosen]
                + startup_incremental_power_fraction
                * incremental_w[chosen]
                * realized_power_factor,
                idle_w[chosen],
                idle_temperature_c[chosen],
                thermal_gain[chosen],
                thermal_tau[chosen],
                evaluation_start_s,
                evaluation_end_s,
            ),
            _thermal_segment_peak(
                actual_temperature_at_service,
                startup_end,
                actual_completion,
                realized_active_w,
                idle_w[chosen],
                idle_temperature_c[chosen],
                thermal_gain[chosen],
                thermal_tau[chosen],
                evaluation_start_s,
                evaluation_end_s,
            ),
        )
        if in_evaluation and max(
            actual_temperature_at_start,
            actual_temperature_at_service,
            actual_finish_temperature,
        ) > thermal_limit_c:
            thermal_violations += 1
        if in_evaluation and actual_derate < 1.0 - 1.0e-12:
            thermal_derate_count += 1

        nominal_activity_energy = incremental_w[chosen] * (
            actual_service + startup_incremental_power_fraction * actual_startup
        )
        nominal_actual_energy = (
            nominal_activity_energy
            + idle_allocation_fraction
            * idle_w[chosen]
            * (actual_completion - now)
        )
        actual_energy = nominal_actual_energy * realized_power_factor
        activity_energy = nominal_activity_energy * realized_power_factor
        charge = candidate_pred_energy[chosen]
        if in_evaluation and actual_energy > charge:
            energy_underestimation_count += 1
            energy_shortfall += actual_energy - charge
        energy_tokens[chosen] -= charge
        energy_debt[chosen] += charge
        window_commitment_j[chosen] += charge
        if in_evaluation:
            max_energy_debt = max(max_energy_debt, energy_debt[chosen])
        if method != 13:
            tenant_energy_tokens[chosen, tenant] -= charge
            tenant_energy_debt[chosen, tenant] += charge
            if in_evaluation:
                max_tenant_energy_debt = max(
                    max_tenant_energy_debt, tenant_energy_debt[chosen, tenant]
                )
        completion_s[request] = actual_completion
        policy_predicted_completion_s[request] = pred_completion
        policy_shift_at_dispatch_s[request] = policy_queue_shift_s[chosen]
        selected_node[request] = chosen
        queue_s[request] = actual_queue
        jitter_s[request] = actual_jitter
        task_energy_j[request] = actual_energy
        activity_energy_j[request] = activity_energy
        charged_energy_j[request] = charge
        if in_evaluation:
            tenant_admitted[tenant] += 1
            node_assignments[chosen] += 1
        tail = completion_tail[chosen]
        if tail < 0:
            completion_head[chosen] = request
        else:
            completion_next[tail] = request
        completion_tail[chosen] = request

    # Drain the remaining completion events without introducing a release.
    while True:
        completion_node = -1
        completion_request = -1
        completion_time = np.inf
        for node in range(nodes):
            head = completion_head[node]
            if head >= 0 and completion_s[head] < completion_time:
                completion_node = node
                completion_request = head
                completion_time = completion_s[head]
        if completion_node < 0:
            break
        completion_head[completion_node] = completion_next[completion_request]
        if completion_head[completion_node] < 0:
            completion_tail[completion_node] = -1
        expected_completion = (
            policy_predicted_completion_s[completion_request]
            + policy_queue_shift_s[completion_node]
            - policy_shift_at_dispatch_s[completion_request]
        )
        prediction_error = completion_time - expected_completion
        policy_queue_shift_s[completion_node] += prediction_error
        pred_device_available[completion_node] = max(
            completion_time,
            pred_device_available[completion_node] + prediction_error,
        )
        for slot in range(cache_slots):
            if (
                pred_cache_service[completion_node, slot] >= 0
                and pred_cache_last[completion_node, slot] >= expected_completion - 1.0e-12
            ):
                pred_cache_last[completion_node, slot] += prediction_error
        pred_temperature_time[completion_node] = max(
            completion_time,
            pred_temperature_time[completion_node] + prediction_error,
        )
        if completion_head[completion_node] < 0:
            pred_device_available[completion_node] = completion_time
            pred_temperature[completion_node] = actual_temperature[completion_node]
            pred_temperature_time[completion_node] = completion_time
            for slot in range(cache_slots):
                pred_cache_service[completion_node, slot] = actual_cache_service[
                    completion_node, slot
                ]
                pred_cache_last[completion_node, slot] = actual_cache_last[
                    completion_node, slot
                ]
                pred_cache_memory[completion_node, slot] = actual_cache_memory[
                    completion_node, slot
                ]
        tokens, token_time = _advance_bucket(
            energy_tokens[completion_node],
            energy_token_time[completion_node],
            completion_time,
            budget_rate[completion_node],
            energy_capacity[completion_node],
        )
        energy_tokens[completion_node] = tokens
        energy_token_time[completion_node] = token_time
        energy_debt[completion_node], energy_debt_time[completion_node] = _advance_debt(
            energy_debt[completion_node],
            energy_debt_time[completion_node],
            completion_time,
            budget_rate[completion_node],
        )
        tenant_done = tenant_by_function[function_id[completion_request]]
        if method != 13:
            tenant_tokens, tenant_token_time = _advance_bucket(
                tenant_energy_tokens[completion_node, tenant_done],
                tenant_energy_token_time[completion_node, tenant_done],
                completion_time,
                budget_rate[completion_node] * tenant_energy_share[tenant_done],
                tenant_energy_capacity[completion_node, tenant_done],
            )
            tenant_energy_tokens[completion_node, tenant_done] = tenant_tokens
            tenant_energy_token_time[completion_node, tenant_done] = tenant_token_time
            (
                tenant_energy_debt[completion_node, tenant_done],
                tenant_energy_debt_time[completion_node, tenant_done],
            ) = _advance_debt(
                tenant_energy_debt[completion_node, tenant_done],
                tenant_energy_debt_time[completion_node, tenant_done],
                completion_time,
                budget_rate[completion_node] * tenant_energy_share[tenant_done],
            )
        deadline_done = _request_deadline(
            completion_request,
            release_s,
            function_id,
            tenant_by_function,
            p99_ms,
            minimum_deadline_s,
            deadline_tail_multiplier,
        )
        completed_in_evaluation = (
            evaluation_start_s <= release_s[completion_request] < evaluation_end_s
        )
        if completion_time - release_s[completion_request] <= deadline_done:
            if completed_in_evaluation:
                tenant_success[tenant_done] += 1

    for node in range(nodes):
        peak_temperature = max(
            peak_temperature,
            _thermal_segment_peak(
                actual_temperature[node],
                actual_temperature_time[node],
                evaluation_end_s,
                idle_w[node],
                idle_w[node],
                idle_temperature_c[node],
                thermal_gain[node],
                thermal_tau[node],
                evaluation_start_s,
                evaluation_end_s,
            ),
        )

    counters = np.asarray(
        [
            cache_evictions_pred,
            cache_evictions_actual,
            false_warm,
            false_cold,
            actual_cold_count,
            predicted_cold_count,
            thermal_violations,
            thermal_derate_count,
            energy_underestimation_count,
        ],
        dtype=np.int64,
    )
    scalars = np.asarray(
        [
            peak_temperature,
            max_energy_debt,
            float(np.sum(energy_debt)),
            max_tenant_energy_debt,
            float(np.sum(tenant_energy_debt)),
            energy_shortfall,
            evaluation_activity_energy,
        ],
        dtype=np.float64,
    )
    return (
        completion_s,
        selected_node,
        rejection_code,
        predicted_cold,
        actual_cold,
        queue_s,
        jitter_s,
        task_energy_j,
        activity_energy_j,
        charged_energy_j,
        tenant_offered,
        tenant_admitted,
        tenant_success,
        node_assignments,
        counters,
        scalars,
    )


@nb.njit(cache=True)
def _receiver_aoi(
    completion_s: np.ndarray,
    release_s: np.ndarray,
    function_id: np.ndarray,
    function_count: int,
) -> np.ndarray:
    admitted = np.nonzero(np.isfinite(completion_s))[0]
    order = admitted[np.argsort(completion_s[admitted])]
    last_generation = np.full(function_count, np.nan, dtype=np.float64)
    peaks = np.full(len(release_s), np.nan, dtype=np.float64)
    for request in order:
        function = function_id[request]
        generation = release_s[request]
        completion = completion_s[request]
        if math.isnan(last_generation[function]):
            last_generation[function] = generation
        elif generation > last_generation[function]:
            peaks[request] = completion - last_generation[function]
            last_generation[function] = generation
    return peaks


def _jain(values: np.ndarray) -> float:
    denominator = len(values) * float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def simulate(
    trace: RequestTrace,
    deployment: Deployment,
    method_name: str,
    scenario: StrictScenario,
    *,
    save_outcomes: Path | None = None,
) -> StrictResult:
    if method_name not in METHODS:
        raise ValueError(f"unknown method: {method_name}")
    if scenario.max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    if scenario.payload_multiplier <= 0:
        raise ValueError("payload_multiplier must be positive")
    if scenario.cache_slot_cap < 0:
        raise ValueError("cache_slot_cap must be nonnegative")
    if scenario.inter_switch_rate_mbps < 0:
        raise ValueError("inter_switch_rate_mbps must be nonnegative")
    if scenario.energy_charge_margin <= 0:
        raise ValueError("energy_charge_margin must be positive")
    if scenario.service_rate_profile not in deployment.service_rate_profiles:
        raise ValueError(
            f"unknown service-rate profile: {scenario.service_rate_profile}"
        )
    if scenario.thermal_gain_multiplier <= 0 or scenario.thermal_derate_multiplier < 0:
        raise ValueError("thermal multipliers must be positive or zero as applicable")
    if scenario.admission_cap_factor < 0:
        raise ValueError("admission_cap_factor must be nonnegative")
    if scenario.deadline_floor_multiplier <= 0 or scenario.deadline_tail_multiplier_factor <= 0:
        raise ValueError("deadline multipliers must be positive")
    if scenario.start_day < 0 or scenario.end_day_exclusive < 0:
        raise ValueError("day boundaries must be nonnegative")
    if scenario.end_day_exclusive and scenario.end_day_exclusive <= scenario.start_day:
        raise ValueError("end_day_exclusive must exceed start_day")
    start_s = scenario.start_day * 86400.0
    end_s = (
        scenario.end_day_exclusive * 86400.0
        if scenario.end_day_exclusive
        else float(trace.release_s[-1]) + 1.0
    )
    start_index = int(np.searchsorted(trace.release_s, start_s, side="left"))
    stop_index = int(np.searchsorted(trace.release_s, end_s, side="left"))
    if scenario.max_requests:
        stop_index = min(stop_index, start_index + scenario.max_requests)
    base_evaluated = stop_index - start_index
    if base_evaluated <= 0:
        raise ValueError("the selected evaluation interval contains no requests")
    effective_end_s = end_s
    if scenario.max_requests:
        effective_end_s = float(np.nextafter(trace.release_s[stop_index - 1], np.inf))
    # State is continuous across the complete pre-evaluation trace.  Requests
    # before start_s warm queues, caches, credits, debts and temperatures but
    # are excluded from every reported metric.
    request_slice = slice(0, stop_index)
    release_s = np.asarray(trace.release_s[request_slice])
    function_id = np.asarray(trace.function_id[request_slice])
    service_z = np.asarray(trace.service_z[request_slice])
    network_z = np.asarray(trace.network_z[request_slice])
    startup_z = np.asarray(trace.startup_z[request_slice])
    (
        release_s,
        function_id,
        service_z,
        network_z,
        startup_z,
        is_attack,
    ) = augment_with_attack(
        release_s,
        function_id,
        service_z,
        network_z,
        startup_z,
        np.asarray(trace.tenant_id),
        scenario,
    )
    evaluation_mask = (release_s >= start_s) & (release_s < effective_end_s)
    evaluated = int(evaluation_mask.sum())
    minutes = max(1, int(math.floor(float(release_s[-1]) / 60.0)) + 1)
    (
        route_links,
        route_length,
        link_rate_bps,
        link_fixed_s,
        link_propagation_s,
        link_background_fraction,
    ) = build_scenario_topology(deployment, scenario)
    background_pred, background_actual = build_background_load(
        deployment, minutes, scenario, link_background_fraction
    )
    service_pressure, service_forecast_sha256 = load_service_pressure(
        deployment.service_forecast_path, minutes
    )
    predicted_power_factor, actual_power_factor = build_power_factors(
        minutes, deployment.port, scenario
    )
    requirement_mask = build_requirement_mask(trace, scenario)
    cache_identity, cache_footprint = build_cache_model(trace, scenario.cache_identity_mode)
    cache_ttl = deployment.cache_ttl_s if scenario.cache_ttl_s is None else scenario.cache_ttl_s
    if cache_ttl < 0:
        raise ValueError("cache TTL must be nonnegative")
    footprint_multiplier = (
        deployment.cache_footprint_multiplier * scenario.cache_footprint_multiplier
    )
    cache_slots = required_cache_slots(
        cache_footprint,
        deployment.usable_memory_mb,
        footprint_multiplier,
        cache_identity,
        scenario.cache_slot_cap,
    )
    admission_cap = int(
        math.floor(deployment.admission_cap_per_tenant * scenario.admission_cap_factor)
    )
    node_indices = np.arange(deployment.nodes)
    port_link = route_links[
        node_indices, route_length.astype(np.int64) - 1
    ]
    capacity_descriptor = deployment.service_rate_profiles[
        scenario.service_rate_profile
    ] * np.sqrt(link_rate_bps[port_link] / 1.0e9)

    started = time.perf_counter()
    output = _simulate_kernel(
        METHODS[method_name],
        release_s,
        start_s,
        effective_end_s,
        function_id,
        service_z,
        network_z,
        startup_z,
        np.asarray(trace.p50_ms),
        np.asarray(trace.p99_ms),
        np.asarray(trace.duration_quantiles_ms),
        np.asarray(trace.duration_quantile_probabilities),
        np.asarray(trace.memory_mb),
        cache_identity,
        cache_footprint,
        np.asarray(trace.tenant_id),
        requirement_mask,
        deployment.capability_mask,
        deployment.usable_memory_mb,
        deployment.service_rate_profiles[scenario.service_rate_profile],
        deployment.idle_w,
        deployment.incremental_w,
        deployment.energy_budget_j_per_s,
        deployment.tenant_energy_share,
        deployment.idle_temperature_c,
        deployment.thermal_gain_c_per_w * scenario.thermal_gain_multiplier,
        deployment.thermal_tau_s,
        deployment.startup_mu_log_s,
        deployment.startup_sigma_log_s,
        route_links,
        route_length,
        link_rate_bps,
        link_fixed_s,
        link_propagation_s,
        background_pred,
        background_actual,
        service_pressure,
        np.asarray(trace.stratum_id),
        capacity_descriptor,
        predicted_power_factor,
        actual_power_factor,
        deployment.network_residual_exponent,
        deployment.network_minimum_capacity_fraction,
        deployment.scd_v_joules,
        deployment.minimum_deadline_s_by_tenant * scenario.deadline_floor_multiplier,
        deployment.deadline_tail_multiplier * scenario.deadline_tail_multiplier_factor,
        deployment.service_prediction_quantile,
        NormalDist().inv_cdf(deployment.startup_prediction_quantile),
        cache_ttl,
        footprint_multiplier,
        cache_slots,
        scenario.payload_multiplier,
        deployment.payload_minimum_bytes,
        deployment.payload_sqrt_memory_coefficient_bytes,
        deployment.startup_incremental_power_fraction,
        scenario.service_variability_multiplier,
        scenario.startup_variability_multiplier,
        scenario.energy_budget_factor,
        float(deployment.raw["controller"]["energy_burst_minutes"]) * 60.0,
        scenario.energy_charge_margin,
        deployment.idle_allocation_fraction,
        admission_cap,
        deployment.admission_window_s,
        int(scenario.admission_guard),
        deployment.thermal_limit_c,
        int(scenario.thermal_guard),
        int(scenario.energy_guard),
        int(scenario.deadline_guard),
        scenario.deadline_guard_multiplier,
        deployment.thermal_derate_start_c,
        deployment.thermal_derate_slope_per_c * scenario.thermal_derate_multiplier,
        deployment.thermal_minimum_service_multiplier,
    )
    (
        completion_s,
        selected_node,
        rejection_code,
        predicted_cold,
        actual_cold,
        queue_s,
        jitter_s,
        task_energy_j,
        activity_energy_j,
        charged_energy_j,
        tenant_offered,
        tenant_admitted,
        _event_tenant_success,
        node_assignments,
        counters,
        scalars,
    ) = output
    aoi_peak_s = _receiver_aoi(
        completion_s, release_s, function_id, len(trace.memory_mb)
    )
    runtime = time.perf_counter() - started

    admitted_mask = (selected_node >= 0) & evaluation_mask
    admitted = int(admitted_mask.sum())
    latency_s = completion_s[admitted_mask] - release_s[admitted_mask]
    admitted_indices = np.nonzero(admitted_mask)[0]
    admitted_function = function_id[admitted_mask]
    admitted_tenant = np.asarray(trace.tenant_id)[admitted_function]
    admitted_day = np.minimum(
        (release_s[admitted_mask] // 86400.0).astype(np.int64), trace.p99_ms.shape[1] - 1
    )
    deadlines = np.maximum(
        deployment.minimum_deadline_s_by_tenant[admitted_tenant]
        * scenario.deadline_floor_multiplier,
        deployment.deadline_tail_multiplier
        * scenario.deadline_tail_multiplier_factor
        * np.asarray(trace.p99_ms)[admitted_function, admitted_day]
        / 1000.0,
    )
    successes = latency_s <= deadlines
    success_count = int(successes.sum())
    tenant_success = np.bincount(
        admitted_tenant[successes], minlength=4
    ).astype(np.int64)
    effective_ceiling_multiplier = (
        scenario.deadline_guard_multiplier if scenario.deadline_guard else 1.0
    )
    ceiling_successes = latency_s <= effective_ceiling_multiplier * deadlines
    ceiling_success_count = int(ceiling_successes.sum())
    admitted_attack = is_attack[admitted_mask].astype(bool)
    attacker_offered = int(is_attack[evaluation_mask].sum())
    attacker_admitted = int(admitted_attack.sum())
    admitted_success_mask = np.zeros(len(release_s), dtype=bool)
    admitted_success_mask[admitted_indices] = successes
    legitimate_offered = evaluated - attacker_offered
    legitimate_success = int(
        np.count_nonzero(admitted_success_mask & evaluation_mask & (is_attack == 0))
    )
    if admitted != int(tenant_admitted.sum()) or admitted != int(node_assignments.sum()):
        raise AssertionError("admission totals disagree across request, tenant and node views")
    if success_count != int(tenant_success.sum()):
        raise AssertionError("deadline success totals disagree with tenant grouping")
    if evaluated - admitted != int(np.count_nonzero(rejection_code[evaluation_mask])):
        raise AssertionError("rejection totals disagree with request outcomes")
    for node in range(deployment.nodes):
        node_completion = completion_s[selected_node == node]
        if np.any(np.diff(node_completion) < -1.0e-12):
            raise AssertionError(f"device completion order moved backward on node {node}")
    tenant_success_ratio = tenant_success / np.maximum(tenant_offered, 1)
    makespan = (
        float(np.nanmax(completion_s[admitted_mask]))
        if admitted
        else float(release_s[evaluation_mask][-1])
    )
    measurement_duration_s = effective_end_s - start_s
    supplied_energy = (
        float(deployment.idle_w.sum()) * measurement_duration_s + float(scalars[6])
    )
    valid_aoi = aoi_peak_s[evaluation_mask & np.isfinite(aoi_peak_s)]

    if save_outcomes is not None:
        save_outcomes.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_outcomes,
            completion_s=completion_s,
            selected_node=selected_node,
            rejection_code=rejection_code,
            predicted_cold=predicted_cold,
            actual_cold=actual_cold,
            queue_s=queue_s,
            jitter_s=jitter_s,
            task_energy_j=task_energy_j,
            activity_energy_j=activity_energy_j,
            charged_energy_j=charged_energy_j,
            aoi_peak_s=aoi_peak_s,
            admitted_request_index=admitted_indices,
            is_attack=is_attack,
            evaluation_mask=evaluation_mask,
        )

    rejection_counts = np.bincount(rejection_code[evaluation_mask], minlength=6)
    return StrictResult(
        method=method_name,
        scenario=scenario.name,
        seed=scenario.seed,
        trace_requests=trace.requests,
        evaluated_requests=evaluated,
        evaluation_start_s=float(start_s),
        evaluation_end_s=float(effective_end_s),
        admitted_requests=admitted,
        rejected_requests=evaluated - admitted,
        on_time_requests=success_count,
        success_ratio=success_count / max(evaluated, 1),
        miss_ratio=1.0 - success_count / max(admitted, 1),
        deadline_ceiling_success_ratio=ceiling_success_count / max(evaluated, 1),
        deadline_ceiling_miss_ratio=1.0 - ceiling_success_count / max(admitted, 1),
        deadline_ceiling_on_time_requests=ceiling_success_count,
        mean_latency_ms=float(np.mean(latency_s) * 1000.0) if admitted else math.nan,
        p95_latency_ms=float(np.quantile(latency_s, 0.95) * 1000.0) if admitted else math.nan,
        p99_latency_ms=float(np.quantile(latency_s, 0.99) * 1000.0) if admitted else math.nan,
        task_energy_mj=float(task_energy_j[admitted_mask].sum(dtype=np.float64) / 1.0e6),
        supplied_energy_mj=supplied_energy / 1.0e6,
        mean_peak_aoi_ms=float(np.mean(valid_aoi) * 1000.0) if valid_aoi.size else math.nan,
        p95_peak_aoi_ms=float(np.quantile(valid_aoi, 0.95) * 1000.0) if valid_aoi.size else math.nan,
        jain_tenant_success=_jain(tenant_success_ratio),
        minimum_tenant_success_ratio=float(np.min(tenant_success_ratio)),
        legitimate_success_ratio=legitimate_success / max(legitimate_offered, 1),
        legitimate_offered_requests=legitimate_offered,
        legitimate_on_time_requests=legitimate_success,
        attacker_offered_requests=attacker_offered,
        attacker_admitted_requests=attacker_admitted,
        attacker_admission_ratio=(
            attacker_admitted / attacker_offered if attacker_offered else math.nan
        ),
        actual_cold_start_ratio=int(counters[4]) / max(admitted, 1),
        predicted_cold_start_ratio=int(counters[5]) / max(admitted, 1),
        actual_cold_start_requests=int(counters[4]),
        predicted_cold_start_requests=int(counters[5]),
        false_warm_ratio=int(counters[2]) / max(admitted, 1),
        false_cold_ratio=int(counters[3]) / max(admitted, 1),
        false_warm_requests=int(counters[2]),
        false_cold_requests=int(counters[3]),
        predicted_cache_evictions=int(counters[0]),
        actual_cache_evictions=int(counters[1]),
        imputed_duration_request_ratio=float(
            np.asarray(trace.duration_imputed)[admitted_function, admitted_day].mean()
        )
        if admitted
        else math.nan,
        cache_identity_mode=scenario.cache_identity_mode,
        cache_slots_allocated=cache_slots,
        admission_cap_per_tenant=admission_cap,
        admission_window_s=deployment.admission_window_s,
        admission_guard=scenario.admission_guard,
        deadline_guard=scenario.deadline_guard,
        deadline_ceiling_multiplier=effective_ceiling_multiplier,
        inter_switch_rate_mbps=scenario.inter_switch_rate_mbps,
        maximum_route_hops=int(route_length.max()),
        rate_rejections=int(rejection_counts[1]),
        energy_rejections=int(rejection_counts[2]),
        compatibility_rejections=int(rejection_counts[3]),
        thermal_rejections=int(rejection_counts[4]),
        deadline_rejections=int(rejection_counts[5]),
        p95_network_queue_ms=float(np.quantile(queue_s[admitted_mask], 0.95) * 1000.0) if admitted else math.nan,
        p95_network_jitter_ms=float(np.quantile(jitter_s[admitted_mask], 0.95) * 1000.0) if admitted else math.nan,
        peak_temperature_c=float(scalars[0]),
        thermal_violations=int(counters[6]),
        thermal_derate_ratio=int(counters[7]) / max(admitted, 1),
        thermal_derated_requests=int(counters[7]),
        energy_underestimation_ratio=int(counters[8]) / max(admitted, 1),
        energy_underestimated_requests=int(counters[8]),
        energy_shortfall_j=float(scalars[5]),
        maximum_per_device_energy_debt_j=float(scalars[1]),
        final_total_device_energy_debt_j=float(scalars[2]),
        maximum_per_device_tenant_energy_debt_j=float(scalars[3]),
        final_total_tenant_energy_debt_j=float(scalars[4]),
        last_completion_s=makespan,
        drain_duration_s=max(0.0, makespan - effective_end_s),
        peak_aoi_samples=int(valid_aoi.size),
        runtime_s=runtime,
        deployment_sha256=deployment.source_sha256,
        simulator_sha256=_file_sha256(Path(__file__)),
        scheduler_core_sha256=_file_sha256(Path(__file__).with_name("scheduler_core.py")),
        service_forecast_sha256=service_forecast_sha256,
        trace_manifest_sha256=_file_sha256(trace.directory / "manifest.json"),
        tenant_offered=[int(value) for value in tenant_offered],
        tenant_admitted=[int(value) for value in tenant_admitted],
        tenant_success=[int(value) for value in tenant_success],
        node_ports=[int(value) for value in deployment.port],
        node_assignments=[int(value) for value in node_assignments],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--deployment",
        type=Path,
        default=Path("config/strict_deployment_20260914.json"),
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="scd")
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--background-multiplier", type=float, default=1.0)
    parser.add_argument("--inter-switch-rate-mbps", type=float, default=0.0)
    parser.add_argument("--cache-ttl-s", type=float)
    parser.add_argument("--cache-footprint-multiplier", type=float, default=1.0)
    parser.add_argument("--cache-identity-mode", choices=("function", "stratum"), default="function")
    parser.add_argument("--cache-slot-cap", type=int, default=0)
    parser.add_argument("--payload-multiplier", type=float, default=1.0)
    parser.add_argument("--power-cv", type=float, default=0.15)
    parser.add_argument("--no-power-margin", action="store_true")
    parser.add_argument("--service-rate-profile", default="batch16")
    parser.add_argument("--thermal-gain-multiplier", type=float, default=1.0)
    parser.add_argument("--thermal-derate-multiplier", type=float, default=1.0)
    parser.add_argument("--admission-cap-factor", type=float, default=1.0)
    parser.add_argument("--no-admission-guard", action="store_true")
    parser.add_argument("--deadline-floor-multiplier", type=float, default=1.0)
    parser.add_argument("--deadline-tail-multiplier-factor", type=float, default=1.0)
    parser.add_argument("--attack-multiplier", type=int, default=1)
    parser.add_argument("--attack-tenant", type=int, default=3)
    parser.add_argument("--energy-budget-factor", type=float, default=1.0)
    parser.add_argument("--accelerator-fraction", type=float, default=0.0)
    parser.add_argument("--include-sensitivity-nodes", action="store_true")
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--end-day-exclusive", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    args = parser.parse_args()
    scenario = StrictScenario(
        name=args.scenario,
        seed=args.seed,
        background_multiplier=args.background_multiplier,
        inter_switch_rate_mbps=args.inter_switch_rate_mbps,
        cache_ttl_s=args.cache_ttl_s,
        cache_footprint_multiplier=args.cache_footprint_multiplier,
        cache_identity_mode=args.cache_identity_mode,
        cache_slot_cap=args.cache_slot_cap,
        payload_multiplier=args.payload_multiplier,
        power_coefficient_of_variation=args.power_cv,
        power_margin_enabled=not args.no_power_margin,
        service_rate_profile=args.service_rate_profile,
        thermal_gain_multiplier=args.thermal_gain_multiplier,
        thermal_derate_multiplier=args.thermal_derate_multiplier,
        admission_cap_factor=args.admission_cap_factor,
        admission_guard=not args.no_admission_guard,
        deadline_floor_multiplier=args.deadline_floor_multiplier,
        deadline_tail_multiplier_factor=args.deadline_tail_multiplier_factor,
        attack_multiplier=args.attack_multiplier,
        attack_tenant=args.attack_tenant,
        energy_budget_factor=args.energy_budget_factor,
        accelerator_fraction=args.accelerator_fraction,
        include_sensitivity_nodes=args.include_sensitivity_nodes,
        start_day=args.start_day,
        end_day_exclusive=args.end_day_exclusive,
        max_requests=args.max_requests,
    )
    trace = load_request_trace(args.trace, verify_hashes=True)
    deployment = load_deployment(
        args.deployment, include_sensitivity_nodes=args.include_sensitivity_nodes
    )
    result = simulate(trace, deployment, args.method, scenario, save_outcomes=args.outcomes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
