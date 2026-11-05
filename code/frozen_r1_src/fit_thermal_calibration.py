"""Chronologically validated first-order thermal fits from complete real blocks.

The fitted intercept is an effective idle CPU temperature, not measured ambient.
PSE output is a system-level predictor, not silicon heat measured by a calorimeter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares


@njit(cache=True)
def thermal_response(times, watts, initial, idle_w, idle_c, gain_c_per_w, tau_s):
    out = np.empty(len(times))
    out[0] = initial
    for i in range(1, len(times)):
        decay = np.exp(-(times[i]-times[i-1])/tau_s)
        target = idle_c+gain_c_per_w*(watts[i-1]-idle_w)
        out[i] = decay*out[i-1]+(1-decay)*target
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    directory = args.directory
    report = json.loads((directory/'analysis.json').read_text())
    phases = [json.loads(line) for line in (directory/'phases.jsonl').read_text().splitlines()]
    origin = phases[0]['start']
    split = next(row['start'] for row in phases if row['block'] == 5)-origin
    thermal = pd.read_csv(directory/'thermal_samples.csv')
    power = pd.read_csv(directory/'power_samples.csv')
    plateaus = pd.read_csv(directory/'plateau_statistics.csv')
    thermal['second'] = np.floor(thermal.time-origin).astype(int)
    # Before/after observations and repeated same sensor values are not independent.
    temperatures = thermal.groupby(['port', 'second'], as_index=False).cpu_c.median()
    fits, sources = [], []
    for board in report['boards']:
        fit = board['achieved_occupancy_fit']
        if not fit or not fit['repeated_active_calibration']:
            continue
        port = board['port']
        own = temperatures[temperatures.port == port].dropna().copy()
        own['time_s'] = own.second+.5
        meter = power[(power.port == port)&power.poe_on].sort_values('time')
        own = own[(own.time_s+origin >= meter.time.min())&(own.time_s+origin <= meter.time.max())]
        own['poe_w'] = np.interp(own.time_s+origin, meter.time, meter.poe_output_w)
        training = own.time_s < split
        train = own[training]
        test = own[~training]
        idle_w = float(plateaus[(plateaus.port == port)&(plateaus.block < 5)&
                                (plateaus.target_duty == 0.)].mean_poe_w.mean())
        times, watts, observed = (train[c].to_numpy(dtype=float) for c in ['time_s', 'poe_w', 'cpu_c'])
        def residual(parameters):
            idle_c, gain, tau = parameters
            return thermal_response(times, watts, observed[0], idle_w, idle_c, gain, tau)-observed
        best = None
        for initial_tau in (10., 60., 300., 1200.):
            candidate = least_squares(residual, [np.percentile(observed, 10), 4., initial_tau],
                                      bounds=([0., 0., 1.], [90., 50., 3600.]),
                                      loss='linear', max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            if best is None or np.sum(candidate.fun**2) < np.sum(best.fun**2):
                best = candidate
        all_times, all_watts = own.time_s.to_numpy(), own.poe_w.to_numpy()
        prediction = thermal_response(all_times, all_watts, own.cpu_c.iloc[0], idle_w, *best.x)
        own['fit_c'] = prediction
        own['partition'] = np.where(training, 'first_five_blocks_fit', 'last_five_blocks_validation')
        # Free-running holdout: no reset at the split and no future temperature input.
        errors = prediction[~training]-test.cpu_c.to_numpy()
        last_training_c = float(train.cpu_c.iloc[-1])
        constant_errors = last_training_c-test.cpu_c.to_numpy()
        fits.append(dict(port=port, training_samples=len(train), validation_samples=len(test),
                         effective_idle_cpu_c=float(best.x[0]), gain_c_per_pse_w=float(best.x[1]),
                         tau_s=float(best.x[2]), idle_power_w=idle_w,
                         optimizer_success=bool(best.success),
                         training_rmse_c=float(np.sqrt(np.mean(best.fun**2))),
                         validation_rmse_c=float(np.sqrt(np.mean(errors**2))),
                         validation_mae_c=float(np.mean(np.abs(errors))),
                         constant_last_training_rmse_c=float(np.sqrt(np.mean(constant_errors**2))),
                         max_validation_abs_error_c=float(np.max(np.abs(errors))),
                         on_parameter_bound=bool(np.any(best.active_mask))))
        sources.append(own)
    pd.concat(sources).to_csv(directory/'thermal_fit_source.csv', index=False)
    result = dict(protocol='First five randomized blocks fit; last five held out chronologically',
                  response='Exact first-order zero-order-hold update: decay*T+(1-decay)*(idle_C+gain*(P-idle_P))',
                  temperature_aggregation='One-second median; controller envelope timestamps',
                  ambient_measurement_available=False,
                  scope='Effective CPU response to PSE supplied power; not a material thermal resistance or certified throttle model',
                  uncertainty='No independent-session CI inferred from autocorrelated one-second residuals',
                  source_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                 for name in ['thermal_samples.csv', 'power_samples.csv', 'plateau_statistics.csv',
                                              'phases.jsonl', 'analysis.json']},
                  boards=fits)
    (directory/'thermal_fit_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
