from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.strict_components import (
    FluidPacketTrainNetwork,
    MemoryBoundLRUCache,
    ThermalState,
    TokenBucket,
)
from src.strict_config import load_deployment
from src.strict_simulate import (
    StrictScenario,
    build_background_load,
    build_power_factors,
    build_scenario_topology,
)
from src.strict_trace import _tenant_from_application, stable_anonymous_id


ROOT = Path(__file__).resolve().parents[1]


def test_cache_checks_execution_start_and_not_release() -> None:
    cache = MemoryBoundLRUCache(capacity_mb=512.0, ttl_s=10.0)
    cache.access(service=1, footprint_mb=100.0, start_s=0.0, completion_s=0.5)
    # A request released at 9 s but starting at 20 s is cold.  This is the
    # counterexample that the earlier aggregate implementation did not model.
    assert cache.preview(service=1, footprint_mb=100.0, start_s=20.0).cold


def test_cache_capacity_is_memory_and_lru_evicts() -> None:
    cache = MemoryBoundLRUCache(capacity_mb=300.0, ttl_s=100.0)
    cache.access(1, 120.0, 0.0, 1.0)
    cache.access(2, 120.0, 2.0, 3.0)
    decision = cache.access(3, 120.0, 4.0, 5.0)
    assert decision.evicted_services == (1,)
    assert set(cache.entries) == {2, 3}
    assert math.isclose(cache.occupied_mb, 240.0)
    assert not cache.preview(2, 120.0, 6.0).cold


def test_cache_rejects_container_larger_than_device_capacity() -> None:
    cache = MemoryBoundLRUCache(capacity_mb=512.0, ttl_s=10.0)
    decision = cache.preview(service=1, footprint_mb=600.0, start_s=0.0)
    assert decision.cold and not decision.feasible


def test_predicted_and_realized_cache_can_diverge() -> None:
    predicted = MemoryBoundLRUCache(capacity_mb=512.0, ttl_s=10.0)
    realized = MemoryBoundLRUCache(capacity_mb=512.0, ttl_s=10.0)
    predicted.access(3, 100.0, 0.0, 1.0)
    realized.access(3, 100.0, 0.0, 1.0)
    predicted_hit = predicted.preview(3, 100.0, 9.0)
    realized_hit = realized.preview(3, 100.0, 20.0)
    assert not predicted_hit.cold
    assert realized_hit.cold


def test_multihop_network_is_store_and_forward() -> None:
    network = FluidPacketTrainNetwork(
        rate_bps=np.asarray([100.0e6, 100.0e6]),
        fixed_s=np.zeros(2),
        propagation_s=np.zeros(2),
        background_fraction=np.zeros(2),
        mtu_bytes=1500,
    )
    preview = network.preview([0, 1], 0.0, 150_000.0)
    one_link_s = 150_000.0 * 8.0 / 100.0e6
    per_hop_jitter = 50.0e-6 + 0.015 * one_link_s
    assert math.isclose(
        preview.completion_s, 2.0 * (one_link_s + per_hop_jitter), rel_tol=1.0e-8
    )


def test_network_commit_preserves_fcfs_queue() -> None:
    network = FluidPacketTrainNetwork(
        rate_bps=np.asarray([8.0e6]),
        fixed_s=np.zeros(1),
        propagation_s=np.zeros(1),
        background_fraction=np.zeros(1),
    )
    first = network.preview([0], 0.0, 1_000_000.0)
    network.commit(first)
    second = network.preview([0], 0.1, 1_000_000.0)
    assert math.isclose(second.queue_s, first.completion_s - 0.1, rel_tol=1.0e-8)


def test_event_driven_thermal_state_cools_and_heats() -> None:
    state = ThermalState(2.0, 6.0, 40.0, 2.0, 10.0, temperature_c=50.0)
    at_start, at_finish = state.commit(start_s=20.0, active_duration_s=10.0)
    assert 40.0 < at_start < 50.0
    assert at_finish > at_start


def test_rejected_event_can_advance_bucket_without_consumption() -> None:
    bucket = TokenBucket(rate_per_s=1.0, capacity=5.0, tokens=0.0)
    bucket.advance(3.0)
    assert bucket.tokens == 3.0
    assert not bucket.consume(4.0, 3.0)
    assert bucket.tokens == 3.0
    assert bucket.consume(4.0, 4.0)


def test_stable_ids_do_not_use_process_salted_hash() -> None:
    assert stable_anonymous_id("app-a") == stable_anonymous_id("app-a")
    assert stable_anonymous_id("app-a") != stable_anonymous_id("app-b")


def test_logical_tenant_is_application_coherent_and_not_trigger_based() -> None:
    app_ids = np.asarray([4, 4, 7, 8], dtype=np.int64)
    tenants = _tenant_from_application(app_ids)
    assert tenants.tolist() == [0, 0, 3, 0]


def test_power_uncertainty_has_unit_mean_and_conservative_predictor() -> None:
    predicted, actual = build_power_factors(
        100_000,
        2,
        StrictScenario(name="power", power_coefficient_of_variation=0.20),
    )
    assert predicted > 1.0
    assert abs(float(actual.mean()) - 1.0) < 0.005
    assert 0.93 < float(np.mean(actual <= predicted)) < 0.97


def test_primary_configuration_uses_fixed_fifteen_board_cohort() -> None:
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    assert deployment.nodes == 15
    assert 4 not in set(deployment.port.tolist())
    assert set(deployment.port.tolist()) == {
        1,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        17,
        19,
        20,
        21,
        22,
        23,
    }


def test_sensitivity_configuration_can_include_port_four() -> None:
    deployment = load_deployment(
        ROOT / "config" / "strict_deployment_20260914.json",
        include_sensitivity_nodes=True,
    )
    assert deployment.nodes == 16
    assert 4 in set(deployment.port.tolist())


def test_optional_topology_adds_one_shared_inter_switch_resource() -> None:
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    topology = build_scenario_topology(
        deployment, StrictScenario(name="two-switch", inter_switch_rate_mbps=100.0)
    )
    routes, lengths, rates = topology[0], topology[1], topology[2]
    assert len(rates) == deployment.links + 1
    assert float(rates[-1]) == 100.0e6
    downstream = set(deployment.simulated_inter_switch_downstream_ports.tolist())
    for node, port in enumerate(deployment.port):
        assert int(lengths[node]) == (3 if int(port) in downstream else 2)
        if int(port) in downstream:
            assert int(routes[node, 1]) == deployment.links


def test_appending_resources_does_not_change_existing_random_streams() -> None:
    deployment = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    scenario = StrictScenario(name="paired", inter_switch_rate_mbps=100.0)
    base_pred, base_actual = build_background_load(deployment, 100, scenario)
    topology = build_scenario_topology(deployment, scenario)
    extended_pred, extended_actual = build_background_load(deployment, 100, scenario, topology[-1])
    assert np.array_equal(base_pred, extended_pred[:, : deployment.links])
    assert np.array_equal(base_actual, extended_actual[:, : deployment.links])

    ids_a = np.asarray([1, 3, 5])
    ids_b = np.asarray([1, 3, 4, 5])
    _, power_a = build_power_factors(100, ids_a, scenario)
    _, power_b = build_power_factors(100, ids_b, scenario)
    assert np.array_equal(power_a[:, 0], power_b[:, 0])
    assert np.array_equal(power_a[:, 1], power_b[:, 1])
    assert np.array_equal(power_a[:, 2], power_b[:, 3])
