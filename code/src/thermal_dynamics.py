"""Exact first-order thermal forcing for nonoverlapping execution intervals."""
from __future__ import annotations
import math
import numpy as np
from numba import njit


@njit(cache=True)
def thermal_step(theta, relative_power, seconds, tau_s):
    decay = math.exp(-max(0., seconds)/tau_s)
    return theta*decay+relative_power*(1.-decay)


@njit(cache=True)
def deposit_thermal_forcing(calendar, node, start, stop, relative_power, tau_s):
    """Exact contribution at each window end, not unnormalized joules/window."""
    if stop <= start:
        return
    first = max(0, int(start//60.))
    last = min(len(calendar)-1, int(stop//60.))
    for window in range(first, last+1):
        left, right = max(start, window*60.), min(stop, (window+1)*60.)
        if right > left:
            end = (window+1)*60.
            calendar[window, node] += relative_power*(
                math.exp(-(end-right)/tau_s)-math.exp(-(end-left)/tau_s))


@njit(cache=True)
def interval_peak(theta_start, relative_power, start, stop, horizon, tau_s):
    """Exact extrema of a constant-power segment, clipped to arrival horizon."""
    if start >= horizon or stop <= start:
        return 0.
    end_value = thermal_step(theta_start, relative_power, min(stop, horizon)-start, tau_s)
    return max(theta_start, end_value)
