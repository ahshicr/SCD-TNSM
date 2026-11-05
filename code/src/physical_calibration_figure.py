"""Real calibration evidence: guarded temperature traces and repeated PSE power.

Preview first; final exports require an explicit --final after visual inspection.
This is not a comparison of scheduling policies or a whole-fleet energy result.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import t


STYLE = {
    1: ('#0072B2', 'o', '-'), 3: ('#D55E00', 's', '--'),
    5: ('#009E73', '^', '-.'), 19: ('#CC79A7', 'D', ':'),
    6: ('#0072B2', 'o', '-'), 7: ('#D55E00', 's', '--'),
    8: ('#CC79A7', 'D', ':'), 9: ('#222222', 'v', '-.'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qa-tools', type=Path)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    phases = [json.loads(x) for x in (args.data/'phases.jsonl').read_text().splitlines()]
    origin = phases[0]['start']
    thermal = pd.read_csv(args.data/'thermal_samples.csv')
    plateau = pd.read_csv(args.data/'plateau_statistics.csv')
    analysis = json.loads((args.data/'analysis.json').read_text())
    complete = [b['port'] for b in analysis['boards'] if b['poe_power_available']
                and b['achieved_occupancy_fit'] is not None
                and b['achieved_occupancy_fit']['repeated_active_calibration']]
    if sorted(complete) != [5, 6, 7, 8, 9]:
        raise ValueError('Changed complete power cohort: review the figure contract')
    thermal = thermal[thermal.port.isin([1, 3, 5, 19])].copy()
    thermal['second'] = np.floor(thermal.time-origin).astype(int)
    thermal = thermal[(thermal.second >= 0)&(thermal.second < 180)]
    trace = thermal.groupby(['port', 'second'], as_index=False).cpu_c.max()
    trace['minutes'] = (trace.second+.5)/60
    trace.to_csv(args.out/'calibration_temperature_source.csv', index=False)
    power = plateau[plateau.port.isin(complete)].copy()
    power.to_csv(args.out/'calibration_power_source.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 6.8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .6,
        'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.dpi': 600})
    fig = plt.figure(figsize=(7.16, 3.78))
    grid = fig.add_gridspec(4, 2, left=.075, right=.985, bottom=.19, top=.90,
                            width_ratios=[1.08, .92], wspace=.30, hspace=.10)
    temp_axes = [fig.add_subplot(grid[row, 0]) for row in range(4)]
    bx = fig.add_subplot(grid[:, 1])
    for row, port in enumerate([1, 3, 5, 19]):
        ax = temp_axes[row]
        for phase in phases[:3]:
            start, end = (phase['start']-origin)/60, (phase['end']-origin)/60
            duty = phase['duty']
            ax.axvspan(start, end,
                       color={0.: '#F4F4F4', .5: '#E6E6E6', 1.: '#D9D9D9'}[duty],
                       zorder=0)
            if row == 0:
                ax.text((start+end)/2, .94, '%d%% duty' % (100*duty),
                        transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.2)
        d = trace[trace.port == port]
        color, marker, line = STYLE[port]
        ax.plot(d.minutes, d.cpu_c, color=color, linestyle='-',
                marker=marker, markersize=2.2, markevery=15)
        if port != 5:
            guard = thermal[(thermal.port == port)&(thermal.cpu_c >= 65)].sort_values('time').iloc[0]
            ax.plot((guard.time-origin)/60, guard.cpu_c, marker='x', ms=6, mew=1.3, color=color)
        ax.axhline(65, color='#555555', lw=.7, ls='--', zorder=1)
        ax.set(xlim=(0, 3), ylim=(48, 68), xticks=[0, 1, 2, 3], yticks=[50, 60, 65])
        ax.text(.02, .13, 'Pi 5, port %d' % port, transform=ax.transAxes,
                color=color, fontsize=6.3, fontweight='bold', va='bottom')
        if row < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time from protocol start (min)')
    temp_axes[1].set_ylabel('CPU temperature (°C)')
    temp_axes[0].set_title('a   Guarded sustained inference by board', loc='left',
                           fontweight='bold', pad=10)
    labels = {5: 'Pi 5 / 5', 6: 'Pi 3 / 6', 7: 'Pi 3 / 7', 8: 'Pi 2 / 8', 9: 'Pi 4 / 9'}
    offsets = (np.arange(len(complete)) - (len(complete)-1)/2) * .145
    hatches = ['', '///', '\\\\', '..', 'xx']
    interval_rows = []
    for port_index, (offset, port) in enumerate(zip(offsets, sorted(complete))):
        color, marker, _ = STYLE[port]
        for x, duty in enumerate([0., .5, 1.]):
            d = power[(power.port == port)&(power.target_duty == duty)].sort_values('block')
            values = d.mean_poe_w.to_numpy()
            if len(values) != 10 or not np.isfinite(values).all():
                raise ValueError('Power plot requires ten complete block means per condition')
            half = float(t.ppf(.975, 9)*values.std(ddof=1)/np.sqrt(10))
            position = x + offset
            bx.bar(position, values.mean(), width=.12, color=color, alpha=.62,
                   edgecolor=color, linewidth=.55, hatch=hatches[port_index],
                   label=labels[port] if x == 0 else None, zorder=2)
            # Deterministic displacement retains all ten rounded block readouts.
            jitter = np.linspace(-.035, .035, 10)
            bx.scatter(position+jitter, values, s=7, facecolors='white',
                       edgecolors=color, marker=marker, alpha=.72,
                       linewidths=.45, zorder=4)
            bx.errorbar(position, values.mean(), yerr=half, fmt='none',
                        color='#303030', lw=.75, capsize=1.8, zorder=5)
            interval_rows.append(dict(port=port, target_duty=duty, blocks=10,
                                      mean_w=float(values.mean()), ci95_half_w=half))
    bx.set(xlim=(-.48, 2.48), ylim=(0, 6.6), xticks=[0, 1, 2],
           xticklabels=['0%', '50%', '100%'], yticks=[0, 2, 4, 6],
           xlabel='Requested single-worker duty', ylabel='PSE supplied power (W)')
    bx.set_title('b   Repeated power calibration', loc='left', fontweight='bold', pad=12)
    bx.legend(loc='upper left', ncol=2, frameon=False,
              columnspacing=.7, handlelength=1.0, borderaxespad=.25)
    axes = temp_axes + [bx]
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(direction='out', length=3, width=.6)
    fig.canvas.draw()
    issues = []
    if args.qa_tools:
        spec = importlib.util.spec_from_file_location('visual_qa', args.qa_tools/'visual_qa.py')
        qa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qa)
        issues = qa.audit_layout(fig)
        qa.print_report(issues)
        if any(level in ('WARN', 'FAIL') for level, _ in issues):
            raise ValueError('Resolve layout audit before publication export')
    renderer = fig.canvas.get_renderer()
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text().strip():
            box = obj.get_window_extent(renderer)
            if min(box.x0, box.y0) < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                raise ValueError('Visible text outside fixed canvas: '+obj.get_text())
    fig.savefig(args.out/'physical_calibration_preview.png', dpi=200)
    pd.DataFrame(interval_rows).to_csv(args.out/'calibration_intervals.csv', index=False)
    audit = dict(backend='Python/matplotlib', width_in=7.16, height_in=3.78,
                 cohort_rule='All boards completing ten active power-calibration blocks',
                 temperature_aggregation='Maximum within each controller one-second bin; first randomized block',
                 temperature_timestamp='Midpoint of each controller send/receive envelope, not synchronized worker time',
                 intervals='Two-sided Student-t 95% across ten randomized blocks',
                 guard_c=65., guard_is_hardware_throttling_threshold=False,
                 final_export_requested=args.final, layout_issues=issues,
                 source_sha256={name: hashlib.sha256((args.data/name).read_bytes()).hexdigest()
                                for name in ['phases.jsonl', 'thermal_samples.csv', 'plateau_statistics.csv', 'analysis.json']})
    if args.final:
        fig.savefig(args.out/'physical_calibration.pdf')
        fig.savefig(args.out/'physical_calibration.svg')
        fig.savefig(args.out/'physical_calibration.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    # Grayscale conversion is generated from the same matplotlib canvas.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    gray = rgba[:, :, :3].astype(float) @ np.array([.2126, .7152, .0722]) / 255.
    plt.imsave(args.out/'physical_calibration_grayscale.png', gray, cmap='gray', vmin=0, vmax=1)
    (args.out/'calibration_figure_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    plt.close(fig)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
