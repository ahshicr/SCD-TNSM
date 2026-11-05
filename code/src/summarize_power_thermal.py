"""Analyze real duty/power plateaus without assuming synchronized board clocks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def overlap_bounds(start, end, duration, window_start, window_end):
    """Compute lies inside controller send/receive envelope; no shared clock needed.

    Return lower/upper possible active duration in a target wall-clock window.
    Assumes board and controller monotonic clocks have equal second units over
    a single job. Timing uncertainty is preserved instead of inventing offsets.
    """
    if duration < -1e-8 or duration > end-start+.005:
        raise ValueError('Worker compute duration exceeds its observed transport envelope')
    duration = max(0., duration)
    inside = max(0., min(end, window_end)-max(start, window_start))
    outside = max(0., end-start-inside)
    return max(0., duration-outside), min(duration, inside)


def interval(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), mean=float(x.mean()),
                ci95_half=float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
                minimum=float(x.min()), maximum=float(x.max()))


def analyze(directory):
    manifest = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    phases = load_lines(directory/'phases.jsonl')
    power = load_lines(directory/'power.jsonl')
    if not manifest.get('all_plateaus_finished') or manifest['power_sampler']['errors']:
        raise ValueError('Sustained collection incomplete; do not produce complete-study statistics')
    power_rows = [dict(time=(s['query_start']+s['query_end'])/2., query_s=s['query_end']-s['query_start'],
                       sample=s['sample'], **port) for s in power for port in s['ports']]
    pd.DataFrame(power_rows).to_csv(directory/'power_samples.csv', index=False)
    raw_power = pd.DataFrame(power_rows)
    reports, all_phases, telemetry_rows = [], [], []
    installed = sorted(set(raw_power.port.tolist()) & (set(manifest['eligible_ports']) |
                                                       {int(x) for x in manifest['excluded_ports']}))
    for port in installed:
        jobs = load_lines(directory/('port_%02d_jobs.jsonl' % port))
        idle = load_lines(directory/('port_%02d_idle.jsonl' % port))
        worker = [b for b in manifest['boards'] if b['port'] == port]
        executed = [j for j in jobs if 'compute_start' in j]
        for job in executed:
            for name in ('before', 'after'):
                tm = job[name]
                telemetry_rows.append(dict(port=port, time=(job['controller_send_start']+job['controller_received'])/2.,
                     observation=name, cpu_c=tm['cpu_c'],
                     frequency_khz=json.dumps(tm['cpufreq_khz'], sort_keys=True),
                     fan_rpm=json.dumps(tm['fans_rpm'], sort_keys=True)))
        for job in idle:
            tm = job['telemetry']
            telemetry_rows.append(dict(port=port, time=(job['controller_send_start']+job['controller_received'])/2.,
                 observation='idle', cpu_c=tm['cpu_c'],
                 frequency_khz=json.dumps(tm['cpufreq_khz'], sort_keys=True),
                 fan_rpm=json.dumps(tm['fans_rpm'], sort_keys=True)))
        port_data = raw_power[raw_power.port == port]
        own_phases = []
        for phase in phases:
            start, end = phase['start']+manifest['settling_seconds'], phase['end']
            samples = port_data[(port_data.time >= start)&(port_data.time < end)]
            energy_lower = energy_upper = 0.
            for job in executed:
                lower, upper = overlap_bounds(job['controller_send_start'], job['controller_received'],
                                               job['compute_end']-job['compute_start'], start, end)
                energy_lower += lower
                energy_upper += upper
            lows, ups = energy_lower/(end-start), energy_upper/(end-start)
            if ups > 1.01:
                raise ValueError('Single-worker achieved occupancy exceeds one')
            temperatures = [row['cpu_c'] for row in telemetry_rows if row['port']==port and start<=row['time']<end]
            temperatures = [x for x in temperatures if x is not None]
            measured = samples[samples.poe_on].poe_output_w.dropna()
            row = dict(port=port, block=phase['block'], phase=phase['phase'],
                       target_duty=phase['duty'], observed_busy_lower=lows, observed_busy_upper=ups,
                       samples_independent=False, power_samples=len(measured),
                       mean_poe_w=float(measured.mean()) if len(measured) else None,
                       min_poe_w=float(measured.min()) if len(measured) else None,
                       max_poe_w=float(measured.max()) if len(measured) else None,
                       mean_cpu_c=float(np.mean(temperatures)) if temperatures else None,
                       max_cpu_c=max(temperatures) if temperatures else None)
            own_phases.append(row)
            all_phases.append(row)
        frame = pd.DataFrame(own_phases)
        measured = frame.dropna(subset=['mean_poe_w'])
        group_stats = {str(duty): interval(part.mean_poe_w.to_numpy())
                       for duty, part in measured.groupby('target_duty')}
        fit = None
        if len(measured) and measured.observed_busy_upper.max() > .05:
            occupancy = (measured.observed_busy_lower.to_numpy()+measured.observed_busy_upper.to_numpy())/2.
            watts = measured.mean_poe_w.to_numpy()
            x = np.column_stack([np.ones(len(occupancy)), occupancy])
            coef, _, rank, _ = np.linalg.lstsq(x, watts, rcond=None)
            if rank == 2:
                residual = watts-x@coef
                active_blocks = int(measured[measured.observed_busy_lower>.1].block.nunique())
                fit = dict(idle_intercept_w=float(coef[0]), active_increment_w=float(coef[1]),
                           predicted_full_active_total_w=float(coef.sum()),
                           rmse_w=float(np.sqrt(np.mean(residual**2))),
                           mean_absolute_residual_w=float(np.mean(np.abs(residual))),
                           maximum_achieved_busy=float(occupancy.max()), active_blocks=active_blocks,
                           repeated_active_calibration=(active_blocks == manifest['blocks']),
                           full_active_is_extrapolated=bool(occupancy.max() < .9),
                           scope='OLS on achieved activity, not requested duty; full-active is a fitted value, not a guaranteed power bound')
        own_telemetry = [x['cpu_c'] for x in telemetry_rows if x['port']==port and x['cpu_c'] is not None]
        guard = next((j for j in jobs if j['status']=='temperature_guard'), None)
        meter = port_data[port_data.poe_on].dropna(subset=['poe_output_w'])
        integrated = None
        if len(meter) >= 2:
            start = max(phases[0]['start'], float(meter.time.min()))
            end = min(phases[-1]['end'], float(meter.time.max()))
            middle = meter[(meter.time>start)&(meter.time<end)]
            sample_time = np.r_[start, middle.time.to_numpy(), end]
            sample_power = np.interp(sample_time, meter.time.to_numpy(), meter.poe_output_w.to_numpy())
            integrated = dict(start=start, end=end, covered_s=end-start,
                              energy_j=float(np.trapezoid(sample_power, sample_time)),
                              interpolation='Linear between timestamped displayed power samples; no extrapolation')
        reports.append(dict(port=port, poe_power_available=bool(port_data.poe_on.any()),
             integrated_poe_energy=integrated,
             successful_jobs=sum(j['status']=='ok' and j['parity_ok'] for j in jobs),
             completed_inferences=sum(j.get('computed', 0) for j in jobs),
             parity_failed=sum(not j.get('parity_ok', False) for j in jobs),
             max_observed_cpu_c=max(own_telemetry) if own_telemetry else None,
             thermal_guard=bool(guard), first_guard_utc=guard['recorded_utc'] if guard else None,
             blocked_before_protocol=str(port) in manifest['excluded_ports'],
             worker_state=worker[0] if worker else None,
             requested_duty_power=group_stats, achieved_occupancy_fit=fit))
    pd.DataFrame(all_phases).to_csv(directory/'plateau_statistics.csv', index=False)
    pd.DataFrame(telemetry_rows).to_csv(directory/'thermal_samples.csv', index=False)
    result = dict(blocks=manifest['blocks'], phases=len(phases),
                  plateau_seconds=manifest['plateau_seconds'], settling_seconds=manifest['settling_seconds'],
                  time_scope='Entire 30-plateau protocol; each fit uses the last 45 s of each 60 s plateau',
                  power_boundary='PSE supplied port power; does not include switch AC input or externally powered boards',
                  power_readout_step_w=.1, accuracy_claim='Display step only; no traceable instrument-accuracy certification',
                  temperature_guard_c=65., power_query_latency_s=interval([s['query_end']-s['query_start'] for s in power]),
                  confidence='Student-t intervals across randomized blocks, not across autocorrelated 1-Hz samples',
                  summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(), boards=reports)
    (directory/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('blocks', 'phases', 'power_boundary')}))
    for row in reports:
        print(json.dumps(dict(port=row['port'], jobs=row['successful_jobs'], max_cpu=row['max_observed_cpu_c'],
                             guard=row['thermal_guard'], power_fit=row['achieved_occupancy_fit'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    analyze(p.parse_args().directory)
