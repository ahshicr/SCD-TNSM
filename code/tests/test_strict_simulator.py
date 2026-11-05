from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.strict_config import load_deployment
from src.strict_simulate import (
    METHODS,
    StrictScenario,
    _advance_debt,
    _cache_commit,
    _interval_overlap,
    _route_preview,
    _receiver_aoi,
    _thermal_segment_peak,
    build_cache_model,
    load_service_pressure,
    required_cache_slots,
    simulate,
)
from src.strict_trace import RequestTrace


ROOT = Path(__file__).resolve().parents[1]


def test_wall_clock_energy_overlap_uses_fixed_half_open_boundary() -> None:
    assert _interval_overlap(5.0, 15.0, 10.0, 20.0) == 5.0
    assert _interval_overlap(10.0, 20.0, 10.0, 20.0) == 10.0
    assert _interval_overlap(0.0, 10.0, 10.0, 20.0) == 0.0
    assert _interval_overlap(20.0, 30.0, 10.0, 20.0) == 0.0


def test_thermal_peak_is_clipped_to_the_fixed_measurement_interval() -> None:
    # This phase cools monotonically from 80 C toward 40 C. The complete-phase
    # peak is 80 C, but the [10, 20) measurement interval begins after cooling.
    peak = _thermal_segment_peak(
        80.0,
        0.0,
        30.0,
        2.0,
        2.0,
        40.0,
        1.0,
        10.0,
        10.0,
        20.0,
    )
    assert np.isclose(peak, 40.0 + 40.0 * np.exp(-1.0))
    assert _thermal_segment_peak(
        80.0,
        0.0,
        5.0,
        2.0,
        2.0,
        40.0,
        1.0,
        10.0,
        10.0,
        20.0,
    ) == -np.inf


def test_receiver_peak_aoi_excludes_first_delivery_and_stale_updates() -> None:
    completion = np.asarray([10.0, 6.0, 8.0, 12.0], dtype=np.float64)
    release = np.asarray([0.0, 1.0, 3.0, 9.0], dtype=np.float64)
    function = np.zeros(4, dtype=np.int32)
    peaks = _receiver_aoi(completion, release, function, 1)
    # Completion order is requests 1, 2, 0, 3. Request 1 initializes the
    # stream, request 2 creates a 7-second pre-delivery peak, request 0 is
    # stale, and request 3 creates a 9-second peak.
    assert np.isnan(peaks[1])
    assert peaks[2] == 7.0
    assert np.isnan(peaks[0])
    assert peaks[3] == 9.0


def make_trace(tmp_path: Path, memory_mb: float = 128.0) -> RequestTrace:
    directory = tmp_path / "trace"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"test": True}), encoding="utf-8")
    release = np.asarray([0.0, 1.0, 2.0, 9.0, 20.0], dtype=np.float64)
    requests = len(release)
    return RequestTrace(
        directory=directory,
        manifest={"test": True},
        release_s=release,
        function_id=np.zeros(requests, dtype=np.int32),
        event_id=np.arange(requests, dtype=np.int64),
        service_z=np.zeros(requests, dtype=np.float32),
        network_z=np.zeros((requests, 4), dtype=np.float32),
        startup_z=np.zeros(requests, dtype=np.float32),
        p50_ms=np.full((1, 14), 50.0, dtype=np.float32),
        p99_ms=np.full((1, 14), 100.0, dtype=np.float32),
        duration_quantiles_ms=np.asarray(
            [[[10.0, 15.0, 30.0, 50.0, 70.0, 100.0, 120.0]] * 14],
            dtype=np.float32,
        ),
        duration_quantile_probabilities=np.asarray(
            [0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0], dtype=np.float64
        ),
        memory_mb=np.asarray([memory_mb], dtype=np.float32),
        tenant_id=np.asarray([0], dtype=np.int8),
        stratum_id=np.asarray([0], dtype=np.int8),
        app_id=np.asarray([1], dtype=np.int64),
        trigger_id=np.asarray([0], dtype=np.int16),
        duration_imputed=np.zeros((1, 14), dtype=np.uint8),
        function_key_digest=np.asarray([7], dtype=np.int64),
    )


def test_request_level_simulator_conserves_all_outcomes(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "scd",
        StrictScenario(name="unit", energy_guard=False, max_requests=5),
    )
    assert result.admitted_requests + result.rejected_requests == 5
    assert sum(result.tenant_offered) == 5
    assert sum(result.node_assignments) == result.admitted_requests
    assert result.actual_cold_start_ratio > 0
    assert result.deadline_ceiling_success_ratio == result.success_ratio
    assert result.deadline_ceiling_miss_ratio == result.miss_ratio
    assert result.evaluation_start_s == 0.0
    assert result.evaluation_end_s == np.nextafter(20.0, np.inf)
    assert result.last_completion_s >= result.evaluation_end_s
    assert result.drain_duration_s == result.last_completion_s - result.evaluation_end_s


def test_deadline_ceiling_retains_strict_and_grace_outcomes_separately(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "scd",
        StrictScenario(
            name="grace",
            energy_guard=False,
            thermal_guard=False,
            deadline_guard=True,
            deadline_guard_multiplier=1000.0,
        ),
    )
    assert result.deadline_guard
    assert result.deadline_ceiling_multiplier == 1000.0
    assert result.deadline_ceiling_on_time_requests >= result.on_time_requests
    assert result.deadline_ceiling_success_ratio >= result.success_ratio
    assert result.deadline_ceiling_miss_ratio <= result.miss_ratio


def test_memory_incompatibility_is_a_hard_constraint(tmp_path: Path) -> None:
    trace = make_trace(tmp_path, memory_mb=8000.0)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "scd",
        StrictScenario(name="memory", energy_guard=False),
    )
    assert result.admitted_requests == 0
    assert result.compatibility_rejections == 5


def test_accelerator_requirement_uses_only_capable_nodes(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "compute_first",
        StrictScenario(
            name="accelerator",
            accelerator_fraction=1.0,
            energy_guard=False,
        ),
    )
    used_ports = {
        port for port, count in zip(result.node_ports, result.node_assignments, strict=True) if count
    }
    assert used_ports
    assert used_ports.issubset({12, 20})


def test_screening_bucket_updates_on_every_release(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "scd",
        StrictScenario(name="screen", admission_cap_factor=0.0, energy_guard=False),
    )
    assert result.rate_rejections == 5
    assert result.admitted_requests == 0


def test_screening_allowance_resets_at_fixed_window_boundary(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "release_s", np.asarray([0.0, 1.0, 2.0, 61.0, 62.0]))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    limited = deployment.__class__(
        **{**deployment.__dict__, "admission_cap_per_tenant": 2}
    )
    result = simulate(
        trace,
        limited,
        "compute_first",
        StrictScenario(name="fixed-window", energy_guard=False, thermal_guard=False),
    )
    assert result.admission_cap_per_tenant == 2
    assert result.admission_window_s == 60.0
    assert result.rate_rejections == 1
    assert result.admitted_requests == 4


def test_cache_slot_allocation_cannot_truncate_memory_feasible_entries() -> None:
    memory = np.asarray([50.0, 100.0, 200.0])
    usable = np.asarray([400.0, 1000.0])
    assert required_cache_slots(memory, usable, 1.0) == 3
    many_services = np.full(100, 50.0)
    assert required_cache_slots(many_services, usable, 1.0) == 20
    assert required_cache_slots(many_services, usable, 1.0, slot_cap=4) == 4


def test_cache_capacity_and_lru_are_applied_at_execution_start() -> None:
    services = np.full(2, -1, dtype=np.int32)
    last = np.zeros(2, dtype=np.float64)
    memory = np.zeros(2, dtype=np.float32)
    assert _cache_commit(1, 100.0, 0.0, 1.0, services, last, memory, 200.0, 10.0) == (1, 0)
    assert _cache_commit(2, 100.0, 2.0, 3.0, services, last, memory, 200.0, 10.0) == (1, 0)
    assert _cache_commit(1, 100.0, 4.0, 5.0, services, last, memory, 200.0, 10.0) == (0, 0)
    assert _cache_commit(3, 100.0, 6.0, 7.0, services, last, memory, 200.0, 10.0) == (1, 1)
    assert set(services.tolist()) == {1, 3}
    assert _cache_commit(4, 100.0, 20.0, 21.0, services, last, memory, 200.0, 10.0) == (1, 0)

    slot_services = np.full(2, -1, dtype=np.int32)
    slot_last = np.zeros(2, dtype=np.float64)
    slot_memory = np.zeros(2, dtype=np.float32)
    _cache_commit(1, 100.0, 0.0, 1.0, slot_services, slot_last, slot_memory, 1000.0, 100.0)
    _cache_commit(2, 100.0, 2.0, 3.0, slot_services, slot_last, slot_memory, 1000.0, 100.0)
    assert _cache_commit(
        3, 100.0, 4.0, 5.0, slot_services, slot_last, slot_memory, 1000.0, 100.0
    ) == (1, 1)
    assert set(slot_services.tolist()) == {2, 3}


def test_multihop_route_waits_for_the_complete_payload() -> None:
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
    assert np.isclose(finish, 2.0301)
    assert queue == 0.0
    assert np.isclose(jitter, 0.0301)


def test_route_uses_cross_traffic_at_each_hop_service_start() -> None:
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
    # The link becomes free at 60 s, so the second background row applies.
    assert queue == 60.0
    assert np.isclose(jitter, 0.35005)
    assert np.isclose(finish, 62.35005)


def test_service_forecast_rejects_future_fitted_origin(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    prediction = np.ones((2, 16), dtype=np.float32)
    scale = np.ones_like(prediction)
    np.savez(path, prediction=prediction, normalization_scale=scale, model_origin=np.asarray([-1, 2]))
    try:
        load_service_pressure(path, 2)
    except ValueError as error:
        assert "future model origin" in str(error)
    else:
        raise AssertionError("future-fitted service forecast was accepted")


def test_held_out_interval_inherits_preceding_cache_state(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "release_s", np.asarray([0.0, 10.0 * 86400.0]))
    object.__setattr__(trace, "function_id", np.zeros(2, dtype=np.int32))
    object.__setattr__(trace, "event_id", np.arange(2, dtype=np.int64))
    object.__setattr__(trace, "service_z", np.zeros(2, dtype=np.float32))
    object.__setattr__(trace, "network_z", np.zeros((2, 4), dtype=np.float32))
    object.__setattr__(trace, "startup_z", np.zeros(2, dtype=np.float32))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "compute_first",
        StrictScenario(
            name="warm-state",
            start_day=10,
            end_day_exclusive=11,
            cache_ttl_s=1.0e6,
            energy_guard=False,
            thermal_guard=False,
        ),
    )
    assert result.evaluated_requests == 1
    assert result.admitted_requests == 1
    assert result.actual_cold_start_ratio == 0.0


def test_strict_method_ids_share_the_manuscript_core() -> None:
    assert METHODS["scd"] == 10
    assert METHODS["dpp_matched"] == 5
    assert METHODS["scd_no_debt"] == 6
    assert METHODS["scd_no_network"] == 7
    assert METHODS["scd_no_forecast"] == 8
    assert METHODS["scd_static_weights"] == 9
    assert METHODS["scd_no_thermal_term"] == 11
    assert METHODS["scd_shared_energy"] == 13


def test_matched_dpp_is_identical_when_scd_deadline_gate_never_excludes(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "p99_ms", np.full((1, 14), 1.0e8, dtype=np.float32))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    deployment = deployment.__class__(
        **{
            **deployment.__dict__,
            "minimum_deadline_s_by_tenant": np.full(4, 1.0e5),
        }
    )
    scd_path = tmp_path / "scd.npz"
    dpp_path = tmp_path / "dpp.npz"
    scenario = StrictScenario(
        name="inactive-gate", energy_guard=False, thermal_guard=False
    )
    simulate(trace, deployment, "scd", scenario, save_outcomes=scd_path)
    simulate(trace, deployment, "dpp_matched", scenario, save_outcomes=dpp_path)
    with np.load(scd_path) as scd, np.load(dpp_path) as dpp:
        for name in (
            "selected_node",
            "completion_s",
            "charged_energy_j",
            "predicted_cold",
            "actual_cold",
        ):
            assert np.array_equal(scd[name], dpp[name], equal_nan=True)


def test_stratum_cache_uses_one_identity_and_conservative_footprint(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "memory_mb", np.asarray([100.0, 250.0], dtype=np.float32))
    object.__setattr__(trace, "stratum_id", np.asarray([3, 3], dtype=np.int8))
    identity, footprint = build_cache_model(trace, "stratum")
    assert identity.tolist() == [3, 3]
    assert footprint.tolist() == [250.0, 250.0]


def test_attack_guard_is_compared_without_changing_its_capacity(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "tenant_id", np.asarray([3], dtype=np.int8))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    limited = deployment.__class__(
        **{**deployment.__dict__, "admission_cap_per_tenant": 2}
    )
    guarded = simulate(
        trace,
        limited,
        "compute_first",
        StrictScenario(
            name="guarded", attack_multiplier=3, energy_guard=False, thermal_guard=False
        ),
    )
    unguarded = simulate(
        trace,
        limited,
        "compute_first",
        StrictScenario(
            name="unguarded",
            attack_multiplier=3,
            admission_guard=False,
            energy_guard=False,
            thermal_guard=False,
        ),
    )
    assert guarded.admission_cap_per_tenant == unguarded.admission_cap_per_tenant == 2
    assert guarded.admitted_requests == 2
    assert guarded.rate_rejections == 13
    assert unguarded.admitted_requests == 15
    assert unguarded.rate_rejections == 0


def test_virtual_energy_debt_uses_joule_credit_between_events() -> None:
    debt, state = _advance_debt(10.0, 0.0, 3.0, 2.0)
    assert debt == 4.0
    assert state == 3.0
    debt, state = _advance_debt(debt, state, 10.0, 2.0)
    assert debt == 0.0
    assert state == 10.0


def test_thermal_derating_is_applied_to_realized_service(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    hot = deployment.__class__(
        **{
            **deployment.__dict__,
            "idle_temperature_c": np.full(deployment.nodes, 64.0),
            "thermal_derate_start_c": 40.0,
        }
    )
    result = simulate(
        trace,
        hot,
        "compute_first",
        StrictScenario(name="thermal-derate", energy_guard=False, thermal_guard=False),
    )
    assert result.thermal_derate_ratio == 1.0


def test_imputed_duration_usage_is_reported(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "duration_imputed", np.ones((1, 14), dtype=np.uint8))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "compute_first",
        StrictScenario(name="imputed", energy_guard=False),
    )
    assert result.imputed_duration_request_ratio == 1.0


def test_synthetic_flood_preserves_legitimate_requests_and_is_auditable(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    object.__setattr__(trace, "tenant_id", np.asarray([3], dtype=np.int8))
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    result = simulate(
        trace,
        deployment,
        "compute_first",
        StrictScenario(
            name="attack",
            attack_multiplier=3,
            energy_guard=False,
        ),
    )
    assert result.evaluated_requests == 15
    assert result.attacker_offered_requests == 10
    assert result.admitted_requests == 15
    assert result.attacker_admitted_requests == 10
