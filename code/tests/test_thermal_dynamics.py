import math
import numpy as np
from src.thermal_dynamics import thermal_step, deposit_thermal_forcing, interval_peak


def test_constant_power_has_declared_equilibrium_not_window_dependent_gain():
    for tau in [4.8, 17., 60., 420.]:
        state = 0.
        for _ in range(2000):
            state = thermal_step(state, 1., 60., tau)
        assert abs(state-1.) < 1e-12


def test_heat_deposit_exact_late_and_early_window_exponential_weighting():
    forcing = np.zeros((3, 1))
    deposit_thermal_forcing(forcing, 0, 90., 150., 1., 60.)
    assert forcing[0, 0] == 0
    assert abs(forcing[1, 0]-(1-math.exp(-.5))) < 1e-12
    assert abs(forcing[2, 0]-(math.exp(-.5)-math.exp(-1))) < 1e-12


def test_segment_partition_invariance_and_constant_power_bound():
    rng = np.random.default_rng(67)
    for tau in [5., 60., 420.]:
        whole = np.zeros((10, 1))
        pieces = whole.copy()
        deposit_thermal_forcing(whole, 0, 0., 600., 1., tau)
        boundaries = np.r_[0., np.sort(rng.uniform(0, 600, 100)), 600.]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            deposit_thermal_forcing(pieces, 0, left, right, 1., tau)
        assert np.allclose(whole, pieces, atol=1e-12)
        assert np.all(whole <= 1.-math.exp(-60/tau)+1e-12)
        state = 0.
        for forcing in whole[:, 0]:
            state = state*math.exp(-60/tau)+forcing
        assert abs(state-thermal_step(0., 1., 600., tau)) < 1e-12


def test_interval_peak_does_not_include_drain_period_or_future_heat():
    assert interval_peak(0., 1., 100., 200., 90., 10.) == 0
    assert abs(interval_peak(0., 1., 100., 200., 110., 10.)-(1-math.exp(-1))) < 1e-12
    assert interval_peak(1., 0., 0., 20., 10., 10.) == 1.


def test_piecewise_overload_never_exceeds_largest_constant_power_equilibrium():
    rng = np.random.default_rng(20260914)
    for tau in [5., 420.]:
        state = 0.
        peak = state
        for _ in range(5000):
            relative_power = rng.uniform(0., 1.75)
            seconds = rng.uniform(0., 3.)
            state = thermal_step(state, relative_power, seconds, tau)
            peak = max(peak, state)
            assert 0. <= state <= 1.75+1e-12
        assert peak > 0.
