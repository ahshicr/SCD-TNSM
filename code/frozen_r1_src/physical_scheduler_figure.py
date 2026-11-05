"""Publication figure for the audited 280-trial live scheduler campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import t


METHODS = ['compute_only', 'energy_greedy', 'dpp_matched', 'scd']
METHOD_LABELS = {
    'compute_only': 'Compute-first', 'energy_greedy': 'Energy-first',
    'dpp_matched': 'Matched DPP', 'scd': 'SCD',
}
COLORS = {
    'compute_only': '#0072B2', 'energy_greedy': '#E69F00',
    'dpp_matched': '#009E73', 'scd': '#CC79A7',
}
SCENARIOS = ['nominal', 'network_pressure', 'energy_pressure']
SCENARIO_LABELS = {
    'nominal': 'Nominal', 'network_pressure': 'Network pressure',
    'energy_pressure': 'Tight energy',
}
METHOD_MARKERS = {'compute_only': 'o', 'energy_greedy': 's',
                  'dpp_matched': '^', 'scd': 'D'}
METHOD_HATCHES = {'compute_only': '', 'energy_greedy': '///',
                  'dpp_matched': '\\\\', 'scd': '..'}
FILLSTYLES = {'nominal': 'full', 'network_pressure': 'none',
              'energy_pressure': 'left'}
ABLATION_COLORS = {
    'scd_no_link': '#0072B2', 'scd_no_debt': '#009E73',
    'scd_no_guard': '#D55E00', 'scd_no_thermal': '#7A6F9B',
}
FAMILY = {
    1: 'Pi 5', 3: 'Pi 5', 5: 'Pi 5', 19: 'Pi 5',
    9: 'Pi 4', 10: 'Pi 4', 17: 'Pi 4', 21: 'Pi 4', 22: 'Pi 4',
    6: 'Pi 3', 7: 'Pi 3', 23: 'Pi 3', 8: 'Pi 2',
    12: 'RDK X5', 20: 'Nano',
}


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise ValueError('Every displayed interval must contain ten finite randomized blocks')
    return float(t.ppf(.975, 9) * values.std(ddof=1) / np.sqrt(10))


def style():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': .65, 'lines.linewidth': 1.0,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.right': False, 'axes.spines.top': False,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })


def validate(trials, ports):
    if len(trials) != 280 or trials['ordinal'].nunique() != 280:
        raise ValueError('The complete 280-trial matrix is required')
    counts = trials.groupby(['cohort', 'scenario', 'method']).size()
    if not (counts == 10).all():
        raise ValueError('Every displayed design cell must contain ten blocks')
    if trials['failed_batches'].sum() or trials['censored_batches'].sum():
        raise ValueError('Unexpected failed or censored physical batches')
    if ports['ordinal'].nunique() != 280 or ports['port'].map(FAMILY).isna().any():
        raise ValueError('Port records do not match the audited hardware-family map')


def make_figure(trials, ports):
    style()
    # Asymmetric quantitative grid.  The PoE result is the primary evidence
    # and receives the wider column.  Each remaining panel uses an encoding
    # matched to its evidence rather than an equal-size scatter template.
    fig = plt.figure(figsize=(7.16, 3.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                            width_ratios=[1.22, .78])
    a_grid = grid[0, 0].subgridspec(1, 2, wspace=.14)
    ax_a_service = fig.add_subplot(a_grid[0, 0])
    ax_a_energy = fig.add_subplot(a_grid[0, 1])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    x_scenarios = np.arange(len(SCENARIOS))
    method_offsets = dict(zip(METHODS, [-.24, -.08, .08, .24]))

    def outcome_bars(ax, frame, metric, scale=1.0, ylim=None):
        width = .18
        for method in METHODS:
            means, errors = [], []
            for scenario in SCENARIOS:
                group = frame[(frame.scenario == scenario) &
                              (frame.method == method)].sort_values('block')
                values = scale * group[metric].to_numpy()
                means.append(values.mean())
                errors.append(ci95(values))
            x = x_scenarios + method_offsets[method]
            ax.bar(x, means, width=width, color=COLORS[method], alpha=.82,
                   edgecolor='white', linewidth=.35,
                   hatch=METHOD_HATCHES[method], zorder=2)
            ax.errorbar(x, means, yerr=errors, fmt='none', color='#303030',
                        capsize=1.4, elinewidth=.65, capthick=.65, zorder=4)
        ax.set_xticks(x_scenarios,
                      [SCENARIO_LABELS[s].replace(' ', '\n') for s in SCENARIOS])
        ax.set_xlim(-.48, len(SCENARIOS) - .52)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis='y', color='#D9D9D9', linewidth=.45, alpha=.8, zorder=0)

    # (a) Grouped bars match panel (b), so all four methods remain visible even
    # when their block means or rounded PSE readings are nearly identical.
    poe = trials[(trials.cohort == 'poe') & trials.method.isin(METHODS)]
    outcome_bars(ax_a_service, poe, 'offered_on_time_success', 100.0, (0, 105))
    outcome_bars(ax_a_energy, poe, 'measured_pse_energy_j', 1.0, (0, 1400))
    ax_a_service.set_ylabel('On-time success (%)')
    ax_a_service.set_title('Service', pad=3)
    ax_a_energy.set_ylabel('PSE energy over 75 s (J)')
    ax_a_energy.set_title('Supplied energy', pad=3)

    # (b) Bars use the full vertical range and make the three operating regimes
    # readable without the large horizontal gaps of a 0--100% forest plot.
    fleet = trials[(trials.cohort == 'fleet') & trials.method.isin(METHODS)]
    width = .18
    for method in METHODS:
        means, errors = [], []
        for scenario in SCENARIOS:
            group = fleet[(fleet.scenario == scenario) &
                          (fleet.method == method)].sort_values('block')
            values = 100 * group.offered_on_time_success.to_numpy()
            means.append(values.mean())
            errors.append(ci95(values))
        x = x_scenarios + method_offsets[method]
        ax_b.bar(x, means, width=width, color=COLORS[method], alpha=.82,
                 edgecolor='white', linewidth=.35,
                 hatch=METHOD_HATCHES[method], zorder=2)
        ax_b.errorbar(x, means, yerr=errors, fmt='none', color='#303030',
                      capsize=1.4, elinewidth=.65, capthick=.65, zorder=4)
    ax_b.set_xticks(x_scenarios,
                    [SCENARIO_LABELS[s].replace(' ', '\n') for s in SCENARIOS])
    ax_b.set_xlim(-.50, len(SCENARIOS) - .50)
    ax_b.set_ylim(0, 105)
    ax_b.set_ylabel('On-time success (%)')
    ax_b.set_title('Fifteen-board service', pad=3)
    ax_b.grid(axis='y', color='#D9D9D9', linewidth=.45, alpha=.8, zorder=0)

    # (c) A zero-centred forest plot displays the paired effect directly.
    # The symmetric-log axis keeps the near-null intervals readable while
    # retaining the much larger service gain obtained by removing the guard.
    ablations = [
        ('network_pressure', 'scd_no_link', 'No link term'),
        ('energy_pressure', 'scd_no_debt', 'No debt term'),
        ('energy_pressure', 'scd_no_guard', 'No energy guard'),
        ('nominal', 'scd_no_thermal', 'No thermal term'),
    ]
    ablation_source = []
    effect_labels = []
    for row, (scenario, method, label) in enumerate(ablations):
        selected = trials[(trials.cohort == 'poe') & (trials.scenario == scenario)
                          & trials.method.isin(['scd', method])]
        wide = selected.pivot(index='block', columns='method',
                              values='offered_on_time_success').sort_index()
        base = 100 * wide['scd'].to_numpy()
        variant = 100 * wide[method].to_numpy()
        delta = variant - base
        color = ABLATION_COLORS[method]
        ax_c.errorbar(delta.mean(), row, xerr=ci95(delta), fmt='o', ms=5.0,
                      color=color, mec='white', mew=.5, capsize=2.0,
                      elinewidth=1.0, zorder=4)
        effect_labels.append(label)
        for block, value in enumerate(delta):
            ablation_source.append(dict(scenario=scenario, method=method,
                                        block=block,
                                        scd_success_percent=base[block],
                                        ablated_success_percent=variant[block],
                                        delta_percentage_points=value))
    ax_c.axvline(0, color='#666666', linestyle=':', linewidth=.8, zorder=1)
    ax_c.set_xscale('symlog', linthresh=1.0, linscale=.9)
    ax_c.set_xlim(-1.6, 120)
    ax_c.set_xticks([-1, 0, 1, 10, 100])
    ax_c.set_yticks(np.arange(len(effect_labels)), effect_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel('Removal minus SCD success (percentage points)')
    ax_c.set_title('Paired physical ablation effects', pad=3)
    ax_c.grid(axis='x', color='#D9D9D9', linewidth=.45, alpha=.8, zorder=0)

    # (d) Full-width composition bars make assignment concentration visible
    # without a heat map.  Closely related low-share boards are combined only
    # in the visual encoding; the exported source retains every family.
    nominal_ports = ports[(ports.cohort == 'fleet') & (ports.scenario == 'nominal')
                          & ports.method.isin(METHODS)].copy()
    nominal_ports['family'] = nominal_ports.port.map(FAMILY)
    per_trial = nominal_ports.groupby(['method', 'block', 'family'], as_index=False).share_of_admitted.sum()
    per_trial['share_percent'] = 100 * per_trial.share_of_admitted
    per_trial['display_family'] = per_trial.family.where(
        per_trial.family.isin(['Pi 5', 'Pi 4']), 'Other')
    display = (per_trial.groupby(['method', 'block', 'display_family'], as_index=False)
               .share_percent.sum())
    family_order = ['Pi 5', 'Pi 4', 'Other']
    family_colors = {'Pi 5': '#356A8A', 'Pi 4': '#E69F00', 'Other': '#B8C2C8'}
    y = np.arange(len(METHODS))
    allocation_source = []
    for row, method in enumerate(METHODS):
        left = 0.0
        for family in family_order:
            values = (display[(display.method == method) &
                              (display.display_family == family)]
                      .sort_values('block').share_percent.to_numpy())
            if len(values) == 0:
                values = np.zeros(10)
            mean = values.mean()
            ax_d.barh(row, mean, left=left, height=.58,
                      color=family_colors[family], edgecolor='white',
                      linewidth=.45, zorder=2)
            allocation_source.extend(
                dict(method=method, block=block, display_family=family,
                     share_percent=value)
                for block, value in enumerate(values))
            left += mean
        pi5_mean = (display[(display.method == method) &
                            (display.display_family == 'Pi 5')]
                    .share_percent.mean())
        pi5_values = (display[(display.method == method) &
                              (display.display_family == 'Pi 5')]
                      .sort_values('block').share_percent.to_numpy())
        ax_d.errorbar(pi5_mean, row, xerr=ci95(pi5_values), fmt='none',
                      ecolor='#202020', elinewidth=.75, capsize=1.8,
                      capthick=.75, zorder=5)
        ax_d.text(pi5_mean / 2, row, f'{pi5_mean:.1f}% Pi 5',
                  color='white', fontsize=6.1, ha='center', va='center',
                  fontweight='bold', zorder=4)
    ax_d.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 100)
    ax_d.set_xticks([0, 25, 50, 75, 100])
    ax_d.set_xlabel('Share of admitted work (%)')
    ax_d.set_title('Nominal assignment composition', pad=3)
    ax_d.grid(axis='x', color='#D9D9D9', linewidth=.45, alpha=.8, zorder=0)
    family_handles = [Patch(facecolor=family_colors[f], edgecolor='none', label=f)
                      for f in family_order]
    ax_d.legend(handles=family_handles, loc='lower right', ncol=3,
                bbox_to_anchor=(1.0, 1.10), fontsize=5.9,
                columnspacing=.8, handlelength=1.1, handletextpad=.3)

    panel_specs = [
        ('(a)', ax_a_service, -.22, 1.08), ('(b)', ax_b, -.22, 1.08),
        ('(c)', ax_c, -.16, 1.08), ('(d)', ax_d, -.22, 1.08),
    ]
    for label, axis, x_label, y_label in panel_specs:
        axis.text(x_label, y_label, label, transform=axis.transAxes, fontsize=8.5,
                  fontweight='bold', va='bottom')
    method_handles = [Line2D([], [], color=COLORS[m], marker=METHOD_MARKERS[m], linestyle='None',
                             markersize=5, label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=method_handles, loc='outside upper center', ncol=4,
                columnspacing=1.3, handletextpad=.35, borderaxespad=.05)
    return fig, pd.DataFrame(ablation_source), per_trial, pd.DataFrame(allocation_source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    trials = pd.read_csv(args.analysis/'trial_records.csv')
    ports = pd.read_csv(args.analysis/'port_records.csv')
    validate(trials, ports)
    fig, ablation_source, per_trial, allocation_source = make_figure(trials, ports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.pdf'))
    fig.savefig(args.out.with_suffix('.svg'))
    fig.savefig(args.out.with_suffix('.tiff'), dpi=600)
    preview = args.out.with_name(args.out.name+'_preview').with_suffix('.png')
    fig.savefig(preview, dpi=200)
    gray = args.out.with_name(args.out.name+'_gray').with_suffix('.png')
    with Image.open(preview) as image:
        image.convert('L').save(gray)
    source = args.out.parent
    trials.to_csv(source/(args.out.name+'_source_trials.csv'), index=False)
    ports.to_csv(source/(args.out.name+'_source_ports.csv'), index=False)
    ablation_source.to_csv(source/(args.out.name+'_source_ablation.csv'), index=False)
    allocation_source.to_csv(source/(args.out.name+'_source_allocation.csv'), index=False)
    concentration = per_trial[per_trial.family == 'Pi 5'].copy()
    concentration['pi5_share_percent'] = concentration.pop('share_percent')
    concentration.to_csv(source/(args.out.name+'_source_pi5_concentration.csv'), index=False)
    print(json.dumps({'trials': len(trials), 'ports': len(ports),
                      'ablation_points': len(ablation_source),
                      'output': str(args.out.with_suffix('.pdf'))}))


if __name__ == '__main__':
    main()
