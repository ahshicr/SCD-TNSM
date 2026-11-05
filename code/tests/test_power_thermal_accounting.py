import numpy as np
from src.summarize_power_thermal import overlap_bounds


def test_activity_bounds_without_clock_synchronization():
    rng = np.random.default_rng(7014)
    for _ in range(5000):
        send = rng.uniform(0, 50)
        receive = send+rng.uniform(.01, 5.)
        duration = rng.uniform(0, receive-send)
        actual_start = rng.uniform(send, receive-duration)
        actual_end = actual_start+duration
        ws = rng.uniform(0, 55)
        we = ws+rng.uniform(.01, 5.)
        truth = max(0., min(actual_end, we)-max(actual_start, ws))
        lo, hi = overlap_bounds(send, receive, duration, ws, we)
        assert lo <= truth+1e-10 <= hi+2e-10


def test_job_entirely_inside_window_has_exact_energy_duration():
    assert overlap_bounds(30., 35., 4., 20., 40.) == (4., 4.)
    assert overlap_bounds(30., 35., 4., 40., 50.) == (0., 0.)
