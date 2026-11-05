from __future__ import annotations

"""Exact small-instance benchmark for the strict online controller.

The production problem is too large for exhaustive optimisation.  This module
enumerates every reject-or-device assignment for a deliberately small request
sequence while replaying the same route, cache, memory and thermal transitions.
It therefore checks the implementation against a genuine finite optimum rather
than against another heuristic.
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from src.strict_components import (
    FluidPacketTrainNetwork,
    MemoryBoundLRUCache,
    ThermalState,
    thermal_advance,
)
from src.strict_config import Deployment, load_deployment
from src.strict_trace import RequestTrace, load_request_trace


@dataclass(frozen=True)
class BenchmarkJob:
    release_s: float
    function_id: int
    service_s: float
    deadline_s: float
    footprint_mb: float
    payload_bytes: float
    requirement_mask: int = 1


@dataclass
class ReplayState:
    network: FluidPacketTrainNetwork
    caches: list[MemoryBoundLRUCache]
    thermal: list[ThermalState]
    device_available_s: np.ndarray


@dataclass(frozen=True)
class AssignmentEvaluation:
    feasible: bool
    objective: float
    successes: int
    rejected: int
    total_energy_j: float
    assignment: tuple[int, ...]


@dataclass(frozen=True)
class OptimalBenchmarkResult:
    jobs: int
    devices: int
    enumerated_assignments: int
    exact_objective: float
    greedy_objective: float
    greedy_gap_percent: float
    exact_successes: int
    greedy_successes: int
    exact_rejections: int
    greedy_rejections: int
    device_ports: list[int]
    exact_assignment_ports: list[int]
    greedy_assignment_ports: list[int]


def make_state(deployment: Deployment) -> ReplayState:
    return ReplayState(
        network=FluidPacketTrainNetwork(
            rate_bps=deployment.link_rate_bps.copy(),
            fixed_s=deployment.link_fixed_s.copy(),
            propagation_s=deployment.link_propagation_s.copy(),
            background_fraction=deployment.link_background_fraction.copy(),
            residual_exponent=deployment.network_residual_exponent,
            minimum_capacity_fraction=deployment.network_minimum_capacity_fraction,
        ),
        caches=[
            MemoryBoundLRUCache(float(capacity), deployment.cache_ttl_s)
            for capacity in deployment.usable_memory_mb
        ],
        thermal=[
            ThermalState(
                float(deployment.idle_w[node]),
                float(deployment.active_w[node]),
                float(deployment.idle_temperature_c[node]),
                float(deployment.thermal_gain_c_per_w[node]),
                float(deployment.thermal_tau_s[node]),
            )
            for node in range(deployment.nodes)
        ],
        device_available_s=np.zeros(deployment.nodes, dtype=np.float64),
    )


def replay_assignment(
    jobs: Sequence[BenchmarkJob],
    deployment: Deployment,
    assignment: Sequence[int],
    *,
    rejection_penalty: float = 8.0,
    energy_weight: float = 0.10,
    cold_weight: float = 0.30,
) -> AssignmentEvaluation:
    if len(jobs) != len(assignment):
        raise ValueError("one assignment is required per job")
    state = make_state(deployment)
    objective = 0.0
    successes = 0
    rejected = 0
    total_energy = 0.0
    for job, node in zip(jobs, assignment, strict=True):
        if node < 0:
            objective += rejection_penalty
            rejected += 1
            continue
        if node >= deployment.nodes:
            return AssignmentEvaluation(False, math.inf, 0, 0, math.inf, tuple(assignment))
        if (deployment.capability_mask[node] & job.requirement_mask) != job.requirement_mask:
            return AssignmentEvaluation(False, math.inf, 0, 0, math.inf, tuple(assignment))
        if job.footprint_mb > deployment.usable_memory_mb[node]:
            return AssignmentEvaluation(False, math.inf, 0, 0, math.inf, tuple(assignment))
        route = deployment.route_links[node, : deployment.route_length[node]]
        network = state.network.preview(
            route,
            job.release_s,
            job.payload_bytes,
            np.zeros(len(route), dtype=np.float64),
        )
        start_s = max(network.completion_s, state.device_available_s[node])
        cache = state.caches[node].preview(job.function_id, job.footprint_mb, start_s)
        if not cache.feasible:
            return AssignmentEvaluation(False, math.inf, 0, 0, math.inf, tuple(assignment))
        startup_s = (
            math.exp(
                deployment.startup_mu_log_s[node]
                + 0.5 * deployment.startup_sigma_log_s[node] ** 2
            )
            if cache.cold
            else 0.0
        )
        thermal = state.thermal[node]
        temperature_start = thermal_advance(
            float(thermal.temperature_c),
            start_s - thermal.state_time_s,
            deployment.idle_w[node],
            deployment.idle_w[node],
            deployment.idle_temperature_c[node],
            deployment.thermal_gain_c_per_w[node],
            deployment.thermal_tau_s[node],
        )
        temperature_service = thermal_advance(
            temperature_start,
            startup_s,
            deployment.idle_w[node]
            + deployment.startup_incremental_power_fraction
            * deployment.incremental_w[node],
            deployment.idle_w[node],
            deployment.idle_temperature_c[node],
            deployment.thermal_gain_c_per_w[node],
            deployment.thermal_tau_s[node],
        )
        derate = max(
            deployment.thermal_minimum_service_multiplier,
            1.0
            - deployment.thermal_derate_slope_per_c
            * max(0.0, temperature_service - deployment.thermal_derate_start_c),
        )
        service_s = job.service_s / (deployment.service_rate[node] * derate)
        duration_s = service_s + startup_s
        temperature_finish = thermal_advance(
            temperature_service,
            service_s,
            deployment.active_w[node],
            deployment.idle_w[node],
            deployment.idle_temperature_c[node],
            deployment.thermal_gain_c_per_w[node],
            deployment.thermal_tau_s[node],
        )
        if max(
            temperature_start,
            temperature_service,
            temperature_finish,
        ) > deployment.thermal_limit_c:
            return AssignmentEvaluation(False, math.inf, 0, 0, math.inf, tuple(assignment))
        completion_s = start_s + duration_s
        latency_s = completion_s - job.release_s
        energy_j = (
            deployment.incremental_w[node]
            * (
                service_s
                + deployment.startup_incremental_power_fraction * startup_s
            )
            + deployment.idle_allocation_fraction
            * deployment.idle_w[node]
            * latency_s
        )
        objective += (
            latency_s / job.deadline_s
            + energy_weight * energy_j / max(deployment.energy_budget_j_per_s[node] * job.deadline_s, 1.0e-9)
            + cold_weight * int(cache.cold)
        )
        successes += int(latency_s <= job.deadline_s)
        total_energy += energy_j
        state.network.commit(network)
        state.caches[node].access(
            job.function_id, job.footprint_mb, start_s, completion_s
        )
        state.thermal[node].temperature_c = temperature_finish
        state.thermal[node].state_time_s = completion_s
        state.device_available_s[node] = completion_s
    return AssignmentEvaluation(
        True,
        objective,
        successes,
        rejected,
        total_energy,
        tuple(int(value) for value in assignment),
    )


def greedy_assignment(
    jobs: Sequence[BenchmarkJob], deployment: Deployment
) -> AssignmentEvaluation:
    assignment: list[int] = []
    for _ in jobs:
        best: AssignmentEvaluation | None = None
        for node in range(-1, deployment.nodes):
            trial = assignment + [node]
            evaluation = replay_assignment(jobs[: len(trial)], deployment, trial)
            if evaluation.feasible and (best is None or evaluation.objective < best.objective):
                best = evaluation
        if best is None:
            raise AssertionError("reject action must always be feasible")
        assignment = list(best.assignment)
    return replay_assignment(jobs, deployment, assignment)


def exact_assignment(
    jobs: Sequence[BenchmarkJob], deployment: Deployment
) -> tuple[AssignmentEvaluation, int]:
    best: AssignmentEvaluation | None = None
    enumerated = 0
    choices = range(-1, deployment.nodes)
    for assignment in itertools.product(choices, repeat=len(jobs)):
        enumerated += 1
        evaluation = replay_assignment(jobs, deployment, assignment)
        if evaluation.feasible and (best is None or evaluation.objective < best.objective):
            best = evaluation
    if best is None:
        raise AssertionError("no feasible assignment found")
    return best, enumerated


def select_deployment(deployment: Deployment, ports: Sequence[int]) -> Deployment:
    indices = [int(np.nonzero(deployment.port == port)[0][0]) for port in ports]
    selected_routes = deployment.route_links[indices]
    selected_lengths = deployment.route_length[indices]
    return Deployment(
        source_path=deployment.source_path,
        source_sha256=deployment.source_sha256,
        raw=deployment.raw,
        node_id=deployment.node_id[indices],
        port=deployment.port[indices],
        device_class=deployment.device_class[indices],
        capability_mask=deployment.capability_mask[indices],
        usable_memory_mb=deployment.usable_memory_mb[indices],
        service_rate=deployment.service_rate[indices],
        service_rate_profiles={
            name: values[indices]
            for name, values in deployment.service_rate_profiles.items()
        },
        idle_w=deployment.idle_w[indices],
        active_w=deployment.active_w[indices],
        incremental_w=deployment.incremental_w[indices],
        energy_budget_j_per_s=deployment.energy_budget_j_per_s[indices],
        tenant_energy_share=deployment.tenant_energy_share,
        idle_temperature_c=deployment.idle_temperature_c[indices],
        thermal_gain_c_per_w=deployment.thermal_gain_c_per_w[indices],
        thermal_tau_s=deployment.thermal_tau_s[indices],
        thermal_limit_c=deployment.thermal_limit_c,
        thermal_derate_start_c=deployment.thermal_derate_start_c,
        thermal_derate_slope_per_c=deployment.thermal_derate_slope_per_c,
        thermal_minimum_service_multiplier=deployment.thermal_minimum_service_multiplier,
        startup_mu_log_s=deployment.startup_mu_log_s[indices],
        startup_sigma_log_s=deployment.startup_sigma_log_s[indices],
        route_links=selected_routes,
        route_length=selected_lengths,
        link_rate_bps=deployment.link_rate_bps,
        link_fixed_s=deployment.link_fixed_s,
        link_propagation_s=deployment.link_propagation_s,
        link_background_fraction=deployment.link_background_fraction,
        network_residual_exponent=deployment.network_residual_exponent,
        network_minimum_capacity_fraction=deployment.network_minimum_capacity_fraction,
        cache_ttl_s=deployment.cache_ttl_s,
        cache_footprint_multiplier=deployment.cache_footprint_multiplier,
        admission_cap_per_tenant=deployment.admission_cap_per_tenant,
        admission_window_s=deployment.admission_window_s,
        deadline_tail_multiplier=deployment.deadline_tail_multiplier,
        service_prediction_quantile=deployment.service_prediction_quantile,
        startup_prediction_quantile=deployment.startup_prediction_quantile,
        scd_v_joules=deployment.scd_v_joules,
        service_forecast_path=deployment.service_forecast_path,
        service_forecast_evidence=deployment.service_forecast_evidence,
        simulated_inter_switch_downstream_ports=deployment.simulated_inter_switch_downstream_ports,
        simulated_inter_switch_propagation_s=deployment.simulated_inter_switch_propagation_s,
        simulated_inter_switch_evidence=deployment.simulated_inter_switch_evidence,
        idle_allocation_fraction=deployment.idle_allocation_fraction,
        payload_minimum_bytes=deployment.payload_minimum_bytes,
        payload_sqrt_memory_coefficient_bytes=deployment.payload_sqrt_memory_coefficient_bytes,
        startup_incremental_power_fraction=deployment.startup_incremental_power_fraction,
        background_ar1_coefficient=deployment.background_ar1_coefficient,
        background_innovation_sd=deployment.background_innovation_sd,
        background_periodic_amplitude=deployment.background_periodic_amplitude,
        background_period_minutes=deployment.background_period_minutes,
        background_burst_period_minutes=deployment.background_burst_period_minutes,
        background_burst_start_minute=deployment.background_burst_start_minute,
        background_burst_duration_minutes=deployment.background_burst_duration_minutes,
        background_burst_amplitude=deployment.background_burst_amplitude,
        background_forecast_alpha=deployment.background_forecast_alpha,
        background_maximum_fraction=deployment.background_maximum_fraction,
        minimum_deadline_s_by_tenant=deployment.minimum_deadline_s_by_tenant,
    )


def jobs_from_trace(trace: RequestTrace, count: int) -> list[BenchmarkJob]:
    if count <= 0 or count > 10:
        raise ValueError("the exact benchmark supports 1 to 10 jobs")
    jobs: list[BenchmarkJob] = []
    origin = float(trace.release_s[0])
    for request in range(count):
        function = int(trace.function_id[request])
        release = float(trace.release_s[request] - origin)
        day = min(int(trace.release_s[request] // 86400.0), trace.p50_ms.shape[1] - 1)
        tenant = int(trace.tenant_id[function])
        service = float(trace.p50_ms[function, day]) / 1000.0
        deadline = max(
            0.15,
            float(trace.p99_ms[function, day]) / 1000.0 * 1.25,
            [0.15, 0.50, 1.00, 2.00][tenant],
        )
        memory = float(trace.memory_mb[function])
        jobs.append(
            BenchmarkJob(
                release_s=release,
                function_id=function,
                service_s=service,
                deadline_s=deadline,
                footprint_mb=memory,
                payload_bytes=max(15_000.0, math.sqrt(memory) * 18_000.0),
            )
        )
    return jobs


def run_benchmark(
    trace: RequestTrace,
    deployment: Deployment,
    *,
    jobs: int = 7,
    ports: Sequence[int] = (1, 9, 23),
) -> OptimalBenchmarkResult:
    small_deployment = select_deployment(deployment, ports)
    small_jobs = jobs_from_trace(trace, jobs)
    exact, enumerated = exact_assignment(small_jobs, small_deployment)
    greedy = greedy_assignment(small_jobs, small_deployment)

    def port_assignment(values: Sequence[int]) -> list[int]:
        return [-1 if node < 0 else int(small_deployment.port[node]) for node in values]

    gap = 100.0 * (greedy.objective - exact.objective) / max(abs(exact.objective), 1.0e-12)
    return OptimalBenchmarkResult(
        jobs=jobs,
        devices=small_deployment.nodes,
        enumerated_assignments=enumerated,
        exact_objective=exact.objective,
        greedy_objective=greedy.objective,
        greedy_gap_percent=gap,
        exact_successes=exact.successes,
        greedy_successes=greedy.successes,
        exact_rejections=exact.rejected,
        greedy_rejections=greedy.rejected,
        device_ports=[int(value) for value in small_deployment.port],
        exact_assignment_ports=port_assignment(exact.assignment),
        greedy_assignment_ports=port_assignment(greedy.assignment),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--deployment",
        type=Path,
        default=Path("config/strict_deployment_20260914.json"),
    )
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--ports", default="1,9,23")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trace = load_request_trace(args.trace, verify_hashes=True)
    deployment = load_deployment(args.deployment)
    ports = tuple(int(value) for value in args.ports.split(","))
    result = run_benchmark(trace, deployment, jobs=args.jobs, ports=ports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
