from __future__ import annotations

"""Generate the four canonical manuscript figures from source-locked runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy import stats


METHOD_ORDER = ["scd", "dpp_matched", "compute_first", "energy_first", "round_robin"]
METHOD_LABEL = {
    "scd": "SCD",
    "dpp_matched": "Matched DPP",
    "compute_first": "Compute-first",
    "energy_first": "Energy-first",
    "round_robin": "Round-robin",
    "scd_no_network": "No network penalty",
    "scd_independent_network": "Independent link state",
}
COLORS = {
    "scd": "#0072B2",
    "dpp_matched": "#D55E00",
    "compute_first": "#009E73",
    "energy_first": "#CC79A7",
    "round_robin": "#4D4D4D",
    "scd_no_network": "#E69F00",
    "scd_independent_network": "#7B3294",
}
MARKERS = {
    "scd": "o",
    "dpp_matched": "s",
    "compute_first": "^",
    "energy_first": "D",
    "round_robin": "P",
    "scd_no_network": "v",
    "scd_independent_network": "X",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
        }
    )


def load_campaign(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = sorted((directory / "runs").glob("*.json"))
    if len(paths) != int(protocol["task_count"]):
        raise AssertionError(
            f"{directory} has {len(paths)} runs but protocol requires {protocol['task_count']}"
        )
    records = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["run_file"] = path.name
        row["run_sha256"] = sha256(path)
        records.append(row)
    frame = pd.DataFrame(records)
    if frame["key"].duplicated().any():
        raise AssertionError(f"duplicate task key in {directory}")
    protocol["protocol_sha256"] = sha256(protocol_path)
    return frame, protocol


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    mean = float(array.mean())
    if len(array) < 2:
        return mean, math.nan, math.nan
    half = float(stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
    return mean, mean - half, mean + half


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=600)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def architecture_figure(output: Path) -> None:
    """Draw the switched data path and causal controller as one closed loop.

    The figure intentionally avoids one connector per board class.  A single
    forward rail represents the request path, while the bottom return rail
    carries only events that have become observable.  This keeps topology and
    causality readable at IEEE two-column width without implying that any
    board class receives a special control path.
    """
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#23313F"
    muted = "#66727E"
    blue = "#0072B2"
    orange = "#D55E00"
    plane_fill = "#F4F6F8"
    control_fill = "#EDF5FA"

    def rounded(x: float, y: float, width: float, height: float, *,
                face: str = "white", edge: str = ink, radius: float = 0.018,
                linewidth: float = 0.85) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            facecolor=face, edgecolor=edge, lw=linewidth,
        ))

    def arrow(start: tuple[float, float], end: tuple[float, float], *,
              color: str = ink, linewidth: float = 1.1,
              style: str = "-|>", connection: str = "arc3,rad=0") -> None:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=8.5, lw=linewidth,
            color=color, connectionstyle=connection, shrinkA=0, shrinkB=0,
        ))

    # The two planes share one visual grammar, but only the control plane is blue.
    rounded(0.018, 0.52, 0.964, 0.42, face=plane_fill, edge="#D5DADF", radius=0.025, linewidth=0.65)
    rounded(0.018, 0.055, 0.964, 0.405, face=control_fill, edge="#C8DCE9", radius=0.025,
            linewidth=0.75)
    ax.text(0.035, 0.905, "REQUEST AND EXECUTION PATH", fontsize=6.7, fontweight="bold",
            color=muted, va="center")
    ax.text(0.035, 0.425, "CAUSAL SCD CONTROL", fontsize=6.7, fontweight="bold",
            color=blue, va="center")

    # Forward data path: aggregate repeated resources instead of drawing six fan-outs.
    rounded(0.050, 0.635, 0.125, 0.175, face="white")
    ax.text(0.1125, 0.770, "Request\nreleases", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=ink, linespacing=0.92)
    ax.text(0.1125, 0.670, "4 tenants", ha="center", va="center", fontsize=6.4, color=muted)
    tenant_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for x, color in zip(np.linspace(0.080, 0.145, 4), tenant_colors, strict=True):
        ax.add_patch(Circle((x, 0.708), 0.009, facecolor=color, edgecolor="none"))

    arrow((0.181, 0.720), (0.226, 0.720))
    rounded(0.232, 0.645, 0.105, 0.150, face="white")
    ax.text(0.2845, 0.764, "Shared\ningress", ha="center", va="center",
            fontsize=6.7, fontweight="bold", color=ink, linespacing=0.92)
    ax.text(0.2845, 0.700, r"FCFS queue $e_0$", ha="center", va="center",
            fontsize=6.2, color=muted)
    for y in (0.660, 0.673, 0.686):
        ax.plot([0.256, 0.313], [y, y], color="#98A2AC", lw=1.05, solid_capstyle="round")

    arrow((0.343, 0.720), (0.380, 0.720))
    rounded(0.386, 0.620, 0.205, 0.200, face="white")
    ax.text(0.4885, 0.787, "Store-and-forward fabric", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=ink)
    for x, label in ((0.435, r"$S_1$"), (0.542, r"$S_2$")):
        ax.add_patch(Polygon(
            [[x - 0.026, 0.712], [x, 0.744], [x + 0.026, 0.712], [x, 0.680]],
            closed=True, facecolor="#E3EFF7", edgecolor=blue, lw=0.9,
        ))
        ax.text(x, 0.712, label, ha="center", va="center", fontsize=7.0, color=ink)
    ax.plot([0.461, 0.516], [0.712, 0.712], color=orange, lw=1.25, ls=(0, (3, 2)))
    ax.text(0.4885, 0.652, "optional inter-switch hop", ha="center", va="center",
            fontsize=6.2, color=orange)

    arrow((0.597, 0.720), (0.628, 0.720))
    rounded(0.634, 0.635, 0.103, 0.175, face="white")
    ax.text(0.6855, 0.768, "Per-board\nFCFS egress", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=ink, linespacing=0.92)
    for y in np.linspace(0.674, 0.714, 4):
        ax.plot([0.657, 0.714], [y, y], color="#98A2AC", lw=1.05, solid_capstyle="round")
    ax.text(0.6855, 0.652, r"$15\times q_e$", ha="center", va="center", fontsize=6.3, color=muted)

    arrow((0.743, 0.720), (0.770, 0.720))
    rounded(0.776, 0.592, 0.176, 0.256, face="white")
    ax.text(0.864, 0.819, "Edge pool", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=ink)
    ax.text(0.864, 0.788, "15 heterogeneous boards", ha="center", va="center",
            fontsize=5.9, color=muted)
    devices = [("Pi 5", "4"), ("Pi 4", "5"), ("Pi 3", "3"),
               ("Pi 2", "1"), ("RDK X5", "1"), ("Nano", "1")]
    for index, (name, count) in enumerate(devices):
        column = index % 2
        row = index // 2
        x = 0.792 + column * 0.078
        y = 0.741 - row * 0.052
        ax.add_patch(Rectangle((x, y - 0.019), 0.068, 0.038,
                               facecolor="#F7F8F9", edgecolor="#B8C0C7", lw=0.55))
        ax.text(x + 0.034, y, f"{name}  {count}", ha="center", va="center",
                fontsize=5.9, color=ink)

    # One numbered cycle replaces the former stack of disconnected text cards.
    stages = [
        (0.130, "Observe", "$q_e, a_i, L_{if}, T_i$\n$Q_i, Z_{ik}$"),
        (0.340, "Predict", "path and execution start\ncache, temperature, energy"),
        (0.550, "Gate", "compatibility and resources\nstrict-deadline feasibility"),
        (0.760, "Select + reserve", "$\\arg\\min_i\\,\\Phi_{ij}$ over eligible set\nreserve links and device"),
    ]
    stage_y = 0.282
    for number, (x, title, subtitle) in enumerate(stages, start=1):
        ax.add_patch(Circle((x, stage_y + 0.045), 0.028, facecolor=blue, edgecolor="white", lw=0.8))
        ax.text(x, stage_y + 0.045, str(number), ha="center", va="center",
                fontsize=7.0, fontweight="bold", color="white")
        ax.text(x, stage_y - 0.005, title, ha="center", va="center",
                fontsize=7.3, fontweight="bold", color=ink)
        ax.text(x, stage_y - 0.061, subtitle, ha="center", va="center",
                fontsize=5.4, color=muted, linespacing=1.12)
    for (x0, _, _), (x1, _, _) in zip(stages[:-1], stages[1:], strict=True):
        arrow((x0 + 0.038, stage_y + 0.045), (x1 - 0.038, stage_y + 0.045),
              color=blue, linewidth=1.15)

    # Dispatch changes reservations immediately, whereas the blue return rail
    # carries only completion and resource-drain events that are now observable.
    arrow((0.760, 0.357), (0.685, 0.628), color=blue, linewidth=1.15,
          connection="arc3,rad=-0.08")
    ax.text(0.748, 0.505, "dispatch + reserve", ha="center", va="center",
            fontsize=6.1, color=blue,
            bbox={"facecolor": plane_fill, "edgecolor": "none", "pad": 0.6})

    feedback_y = 0.095
    ax.plot([0.065, 0.902], [feedback_y, feedback_y], color=blue, lw=1.15,
            solid_capstyle="round")
    arrow((0.902, 0.587), (0.902, feedback_y), color=blue, linewidth=1.15)
    ax.plot([0.065, 0.065], [feedback_y, stage_y + 0.045], color=blue, lw=1.15)
    arrow((0.065, stage_y + 0.045), (0.100, stage_y + 0.045), color=blue, linewidth=1.15)
    ax.text(0.690, feedback_y + 0.022, "completion and resource-drain events",
            ha="center", va="bottom", fontsize=6.1, color=blue,
            bbox={"facecolor": control_fill, "edgecolor": "none", "pad": 0.5})
    save_figure(fig, output, "fig1_architecture")


def raw_dot_ci(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    value: str,
    xlabel: str,
    *,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    zero: bool = False,
) -> None:
    y = np.arange(len(labels), dtype=float)
    colors = colors or ["#0072B2"] * len(labels)
    markers = markers or ["o"] * len(labels)
    for row, label in enumerate(labels):
        values = frame.loc[frame["plot_label"] == label, value].to_numpy(dtype=float)
        offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, row + offsets, s=12, facecolors="white", edgecolors=colors[row],
                   marker=markers[row], linewidths=0.7, zorder=3)
        mean, lower, upper = mean_ci(values)
        if np.isfinite(lower):
            ax.errorbar(mean, row, xerr=[[mean - lower], [upper - mean]], fmt=markers[row],
                        color=colors[row], markerfacecolor=colors[row], markeredgecolor="white",
                        markeredgewidth=0.45, markersize=4.8, capsize=2.0, lw=1.0, zorder=4)
        else:
            ax.scatter([mean], [row], s=22, color=colors[row], marker=markers[row], zorder=4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    if zero:
        ax.axvline(0.0, color="#333333", lw=0.7)


def horizontal_mean_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    value: str,
    xlabel: str,
    *,
    colors: list[str],
    markers: list[str],
    digits: int,
    zero_centered: bool = False,
) -> None:
    """Compact bars with raw seeds and Student-t intervals."""
    y = np.arange(len(labels), dtype=float)
    summaries: list[tuple[float, float, float]] = []
    all_values: list[float] = []
    for label in labels:
        values = frame.loc[frame.plot_label == label, value].to_numpy(dtype=float)
        summaries.append(mean_ci(values))
        all_values.extend(values.tolist())
    means = np.asarray([item[0] for item in summaries])
    lower = np.asarray([item[1] for item in summaries])
    upper = np.asarray([item[2] for item in summaries])
    ax.barh(
        y,
        means,
        height=0.62,
        color=colors,
        alpha=0.78,
        edgecolor="#333333",
        linewidth=0.55,
        zorder=1,
    )
    for row, label in enumerate(labels):
        values = frame.loc[frame.plot_label == label, value].to_numpy(dtype=float)
        offsets = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            values,
            row + offsets,
            s=13,
            facecolors="white",
            edgecolors=colors[row],
            marker=markers[row],
            linewidths=0.75,
            zorder=4,
        )
        mean, lo, hi = summaries[row]
        if np.isfinite(lo):
            ax.errorbar(
                mean,
                row,
                xerr=[[mean - lo], [hi - mean]],
                fmt="none",
                ecolor="#202020",
                elinewidth=0.9,
                capsize=2.2,
                zorder=5,
            )
    finite_bounds = [*all_values, *lower[np.isfinite(lower)], *upper[np.isfinite(upper)], 0.0]
    low = float(min(finite_bounds))
    high = float(max(finite_bounds))
    span = max(high - low, 1.0e-9)
    margin = 0.055 * span
    if zero_centered:
        ax.set_xlim(low - 0.15 * span, high + 0.20 * span)
        ax.axvline(0.0, color="#202020", lw=0.8, zorder=3)
    else:
        ax.set_xlim(0.0, high + 0.20 * span)
    for row, mean in enumerate(means):
        text_color = "#202020"
        if zero_centered and mean < 0 and abs(mean) > 0.25 * span:
            x, ha = mean + margin, "left"
            text_color = "white"
        elif zero_centered and mean < 0:
            x, ha = mean - margin, "right"
        else:
            x, ha = mean + margin, "left"
        ax.text(
            x,
            row,
            f"{mean:.{digits}f}",
            ha=ha,
            va="center",
            fontsize=6.2,
            color=text_color,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#E2E2E2", lw=0.45)
    ax.tick_params(axis="y", length=0)


def grouped_percent_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    display_labels: list[str],
    xlabel: str,
) -> None:
    """Show offered success and cold-start rate on their common percentage scale."""
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    specs = (
        ("success_ratio", "Offered success", COLORS["scd"], ""),
        ("actual_cold_start_ratio", "Cold starts", COLORS["dpp_matched"], "///"),
    )
    for column, (metric, _, color, hatch) in enumerate(specs):
        positions = x + (column - 0.5) * width
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        groups: list[np.ndarray] = []
        for label in labels:
            values = 100.0 * frame.loc[frame.plot_label == label, metric].to_numpy(dtype=float)
            groups.append(values)
            mean, low, high = mean_ci(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        bars = ax.bar(
            positions,
            means,
            width=width * 0.88,
            color=color,
            alpha=0.82,
            edgecolor="#333333",
            linewidth=0.55,
            hatch=hatch,
            zorder=2,
        )
        for position, values, mean, low, high in zip(
            positions, groups, means, lows, highs, strict=True
        ):
            offsets = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(
                position + offsets,
                values,
                s=10,
                facecolors="white",
                edgecolors=color,
                linewidths=0.65,
                zorder=4,
            )
            if np.isfinite(low):
                ax.errorbar(
                    position,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor="#202020",
                    elinewidth=0.8,
                    capsize=2.0,
                    zorder=5,
                )
        for bar, mean in zip(bars, means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 2.0,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
            )
    ax.set_xticks(x, display_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Requests (%)")
    ax.set_ylim(0.0, 104.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#E2E2E2", lw=0.45)


def primary_figure(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = frame[(frame.family == "primary") & (frame.scenario == "primary")].copy()
    data = data[data.method.isin(METHOD_ORDER)]
    data["plot_label"] = data.method.map(METHOD_LABEL)
    data["failure_per_1000"] = 1000.0 * (1.0 - data.success_ratio)
    data["energy_per_offered_j"] = data.task_energy_mj * 1.0e6 / data.evaluated_requests
    data["mean_latency_s"] = data.mean_latency_ms / 1000.0
    data["jain_shortfall_per_1000"] = 1000.0 * (1.0 - data.jain_tenant_success)
    ordered_labels = [METHOD_LABEL[item] for item in METHOD_ORDER]
    colors = [COLORS[item] for item in METHOD_ORDER]
    markers = [MARKERS[item] for item in METHOD_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.45), constrained_layout=True)
    settings = [
        ("failure_per_1000", "Offered failures per 1000", 1),
        ("energy_per_offered_j", "Task energy (J/offered request)", 3),
        ("mean_latency_s", "Mean admitted response (s)", 2),
        ("jain_shortfall_per_1000", r"Tenant unfairness $1000(1-J)$", 2),
    ]
    for ax, (value, xlabel, digits), label in zip(
        axes.flat, settings, ("(a)", "(b)", "(c)", "(d)"), strict=True
    ):
        horizontal_mean_bars(
            ax,
            data,
            ordered_labels,
            value,
            xlabel,
            colors=colors,
            markers=markers,
            digits=digits,
        )
        panel_label(ax, label)
    save_figure(fig, output, "fig2_primary")
    return data


def sensitivity_figure(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    def add(rows: pd.DataFrame, label: str, section: str) -> None:
        value = rows.copy()
        value["plot_label"] = label
        value["section"] = section
        pieces.append(value)

    add(frame[(frame.family == "primary") & (frame.scenario == "primary") & (frame.method == "scd")],
        "Binomial, uniform", "sampling")
    add(frame[(frame.family == "sampling") & (frame.scenario == "unbiased_rounding") & (frame.method == "scd")],
        "Unbiased rounding", "sampling")
    add(frame[(frame.family == "arrival_reconstruction") & (frame.scenario == "stratified_within_minute") & (frame.method == "scd")],
        "Binomial, stratified", "sampling")
    add(frame[(frame.family == "sampling") & (frame.scenario == "deterministic_rounding") & (frame.method == "scd")],
        "Deterministic rounding", "sampling")
    for slots in (1, 2, 4, 8, 16):
        add(frame[(frame.family == "cache_capacity") & (frame.scenario == f"cache_slots_{slots}") & (frame.method == "scd")],
            "1 slot" if slots == 1 else f"{slots} slots", "capacity")
    add(frame[(frame.family == "primary") & (frame.scenario == "primary") & (frame.method == "scd")],
        "Memory limit", "capacity")
    add(frame[(frame.family == "cache_ttl") & (frame.scenario == "ttl_30") & (frame.method == "scd")],
        "30 s", "ttl")
    add(frame[(frame.family == "primary") & (frame.scenario == "primary") & (frame.method == "scd")],
        "300 s", "ttl")
    add(frame[(frame.family == "cache_ttl") & (frame.scenario == "ttl_900") & (frame.method == "scd")],
        "900 s", "ttl")
    data = pd.concat(pieces, ignore_index=True)
    data["failure_per_1000"] = 1000.0 * (1.0 - data.success_ratio)
    data["cold_per_1000"] = 1000.0 * data.actual_cold_start_ratio
    fig = plt.figure(figsize=(7.16, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.10, 1.15, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    axes = [
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Patch(facecolor=COLORS["scd"], edgecolor="#333333", alpha=0.82,
                  label="Offered success"),
            Patch(facecolor=COLORS["dpp_matched"], edgecolor="#333333", alpha=0.82,
                  hatch="///", label="Cold starts"),
        ],
        loc="center",
        ncol=2,
        frameon=False,
    )
    sampling_labels = ["Binomial, uniform", "Unbiased rounding", "Binomial, stratified", "Deterministic rounding"]
    grouped_percent_bars(
        axes[0],
        data[data.section == "sampling"],
        sampling_labels,
        ["Binomial\nuniform", "Unbiased\nrounding", "Binomial\nstratified", "Deterministic\nrounding"],
        "Arrival reconstruction",
    )
    capacity_labels = ["1 slot", "2 slots", "4 slots", "8 slots", "16 slots", "Memory limit"]
    grouped_percent_bars(
        axes[1],
        data[data.section == "capacity"],
        capacity_labels,
        ["1", "2", "4", "8", "16", "Memory"],
        "Cache capacity (entries or byte limit)",
    )
    ttl_labels = ["30 s", "300 s", "900 s"]
    grouped_percent_bars(
        axes[2],
        data[data.section == "ttl"],
        ttl_labels,
        ["30", "300", "900"],
        "Cache TTL (s)",
    )
    for ax, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        panel_label(ax, label)
    save_figure(fig, output, "fig3_sampling_cache")
    return data


def paired_effects(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    challengers: list[tuple[str, pd.DataFrame]],
    metric: str,
    scale: float,
) -> pd.DataFrame:
    rows = []
    for label, challenger in challengers:
        pair = reference[["seed", metric]].merge(
            challenger[["seed", metric]], on="seed", suffixes=("_reference", "_challenger")
        )
        for item in pair.itertuples(index=False):
            rows.append(
                {
                    "plot_label": label,
                    "seed": int(item.seed),
                    "effect": scale * (
                        float(getattr(item, f"{metric}_challenger"))
                        - float(getattr(item, f"{metric}_reference"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def effects_figure(
    main: pd.DataFrame,
    network: pd.DataFrame,
    deadline: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    primary = main[(main.family == "primary") & (main.scenario == "primary")]
    scd = primary[primary.method == "scd"]
    challenger_order = ["dpp_matched", "compute_first", "energy_first", "round_robin"]
    challengers = [(METHOD_LABEL[item], primary[primary.method == item]) for item in challenger_order]
    success = paired_effects(main, scd, challengers, "success_ratio", 100.0)

    energy_source = primary.copy()
    energy_source["energy_per_offered_j"] = (
        energy_source.task_energy_mj * 1.0e6 / energy_source.evaluated_requests
    )
    energy_ref = energy_source[energy_source.method == "scd"]
    energy_challengers = [
        (METHOD_LABEL[item], energy_source[energy_source.method == item]) for item in challenger_order
    ]
    energy = paired_effects(
        energy_source, energy_ref, energy_challengers, "energy_per_offered_j", 1.0
    )

    network_rows = []
    condition_labels = {
        "stress_100mbps_bg12_payload4": r"100 Mb/s, $4\times$ payload",
        "stress_10mbps_bg12_payload4": r"10 Mb/s, $4\times$ payload",
        "stress_10mbps_bg12_payload16": r"10 Mb/s, $16\times$ payload",
    }
    for scenario, condition_label in condition_labels.items():
        subset = network[network.scenario == scenario]
        reference = subset[subset.method == "scd"]
        for method in ("scd_no_network", "scd_independent_network"):
            challenger = subset[subset.method == method]
            effect = paired_effects(network, reference, [(METHOD_LABEL[method], challenger)],
                                    "success_ratio", 100.0)
            effect["condition"] = condition_label
            effect["method"] = method
            network_rows.append(effect)
    network_effect = pd.concat(network_rows, ignore_index=True)

    queue_rows = []
    for scenario, condition_label in condition_labels.items():
        base = network[(network.scenario == scenario) & (network.method == "scd")].copy()
        base["mode"] = "Best effort"
        strict = deadline[
            (deadline.scenario == f"{scenario}_strict") & (deadline.method == "scd")
        ].copy()
        strict["mode"] = "Strict reject"
        grace = deadline[
            (deadline.scenario == f"{scenario}_grace4") & (deadline.method == "scd")
        ].copy()
        grace["mode"] = r"$4\times$ ceiling"
        for rows in (base, grace, strict):
            rows["condition"] = condition_label
            queue_rows.append(rows)
    queue = pd.concat(queue_rows, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.75), constrained_layout=True)
    labels = [METHOD_LABEL[item] for item in challenger_order]
    colors = [COLORS[item] for item in challenger_order]
    markers = [MARKERS[item] for item in challenger_order]
    horizontal_mean_bars(
        axes[0, 0],
        success,
        labels,
        "effect",
        "Success difference (percentage points)",
        colors=colors,
        markers=markers,
        digits=2,
        zero_centered=True,
    )
    horizontal_mean_bars(
        axes[0, 1],
        energy,
        labels,
        "effect",
        "Task-energy difference (J/offered request)",
        colors=colors,
        markers=markers,
        digits=2,
        zero_centered=True,
    )

    ax = axes[1, 0]
    conditions = list(condition_labels.values())
    y = np.arange(len(conditions), dtype=float)
    network_values = network_effect.effect.to_numpy(dtype=float)
    network_low = float(min(network_values.min(), 0.0))
    network_high = float(max(network_values.max(), 0.0))
    network_span = max(network_high - network_low, 1.0e-9)
    for method, offset in (("scd_no_network", -0.12), ("scd_independent_network", 0.12)):
        for row, condition in enumerate(conditions):
            values = network_effect[
                (network_effect.condition == condition) & (network_effect.method == method)
            ].effect.to_numpy(dtype=float)
            mean, lower, upper = mean_ci(values)
            ax.barh(
                row + offset,
                mean,
                height=0.20,
                color=COLORS[method],
                alpha=0.80,
                edgecolor="#333333",
                linewidth=0.5,
                zorder=2,
            )
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(values, row + offset + jitter, s=10, facecolors="white",
                       edgecolors=COLORS[method], marker=MARKERS[method], linewidths=0.7)
            ax.errorbar(mean, row + offset, xerr=[[mean - lower], [upper - mean]],
                        fmt=MARKERS[method], color=COLORS[method], markerfacecolor=COLORS[method],
                        markeredgecolor="white", markeredgewidth=0.4, markersize=4.5, capsize=2)
    ax.set_yticks(y, conditions)
    ax.invert_yaxis()
    ax.set_ylim(len(conditions) - 0.5, -1.05)
    ax.set_xlim(network_low - 0.18 * network_span, network_high + 0.15 * network_span)
    ax.axvline(0.0, color="#333333", lw=0.7)
    ax.grid(axis="x")
    ax.set_xlabel("Success difference from full SCD (percentage points)")
    ax.legend(
        handles=[
            Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="none",
                   label=METHOD_LABEL[m], markersize=4.5)
            for m in ("scd_no_network", "scd_independent_network")
        ],
        loc="upper center", ncol=2, frameon=False,
    )

    ax = axes[1, 1]
    mode_style = {
        "Best effort": ("#4D4D4D", "o", ""),
        r"$4\times$ ceiling": ("#E69F00", "s", "//"),
        "Strict reject": ("#0072B2", "^", "xx"),
    }
    for mode, offset in (("Best effort", -0.16), (r"$4\times$ ceiling", 0.0), ("Strict reject", 0.16)):
        color, marker, hatch = mode_style[mode]
        for row, condition in enumerate(conditions):
            values = queue[(queue.condition == condition) & (queue["mode"] == mode)].p95_network_queue_ms.to_numpy(dtype=float)
            mean, lower, upper = mean_ci(values)
            positive = np.maximum(values, 1.0e-3)
            mean_plot = max(mean, 1.0e-3)
            ax.barh(
                row + offset,
                mean_plot - 1.0e-3,
                left=1.0e-3,
                height=0.25,
                color=color,
                alpha=0.78,
                edgecolor="#333333",
                linewidth=0.45,
                hatch=hatch,
                zorder=2,
            )
            jitter = np.linspace(-0.03, 0.03, len(positive))
            ax.scatter(positive, row + offset + jitter, s=10, facecolors="white",
                       edgecolors=color, marker=marker, linewidths=0.7)
            if np.isfinite(lower):
                lower_plot = max(lower, 1.0e-3)
                upper_plot = max(upper, mean_plot)
                ax.errorbar(mean_plot, row + offset,
                            xerr=[[mean_plot - lower_plot], [upper_plot - mean_plot]],
                            fmt=marker, color=color, markerfacecolor=color, markeredgecolor="white",
                            markeredgewidth=0.4, markersize=4.5, capsize=2)
    ax.set_xscale("log")
    ax.set_xlim(5.0e-4, 3.0e5)
    ax.set_yticks(y, conditions)
    ax.invert_yaxis()
    ax.set_ylim(len(conditions) - 0.5, -1.05)
    ax.grid(axis="x", which="both")
    ax.set_xlabel("P95 network queue waiting (ms, log scale)")
    ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#333333", hatch=hatch, alpha=0.78, label=mode)
            for mode, (color, _, hatch) in mode_style.items()
        ],
        loc="upper center", ncol=3, frameon=False,
    )
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)"), strict=True):
        panel_label(ax, label)
    save_figure(fig, output, "fig4_effects_stress")

    success["panel"] = "4a_success"
    energy["panel"] = "4b_energy"
    network_effect["panel"] = "4c_network"
    queue["panel"] = "4d_queue"
    return pd.concat(
        [success, energy, network_effect, queue], ignore_index=True, sort=False
    )


def load_scaling_figure(
    main: pd.DataFrame,
    load: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    nominal = main[(main.family == "primary") & (main.scenario == "primary")].copy()
    nominal["load_factor"] = 1.0
    lower = load[load.scenario == "load_0p5"].copy()
    lower["load_factor"] = 0.5
    upper = load[load.scenario == "load_2"].copy()
    upper["load_factor"] = 2.0
    data = pd.concat([lower, nominal, upper], ignore_index=True, sort=False)
    data = data[data.method.isin(("scd", "dpp_matched", "compute_first"))].copy()
    data["plot_label"] = data.apply(
        lambda row: rf"{row.load_factor:g}$\times$  {METHOD_LABEL[row.method]}", axis=1
    )
    data["failure_per_1000"] = 1000.0 * (1.0 - data.success_ratio)
    data["energy_per_offered_j"] = data.task_energy_mj * 1.0e6 / data.evaluated_requests
    labels = [
        rf"{factor:g}$\times$  {METHOD_LABEL[method]}"
        for factor in (0.5, 1.0, 2.0)
        for method in ("scd", "dpp_matched", "compute_first")
    ]
    colors = [COLORS[method] for _ in (0.5, 1.0, 2.0) for method in ("scd", "dpp_matched", "compute_first")]
    markers = [MARKERS[method] for _ in (0.5, 1.0, 2.0) for method in ("scd", "dpp_matched", "compute_first")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.05), constrained_layout=True)
    raw_dot_ci(
        axes[0], data, labels, "failure_per_1000", "Offered failures per 1000",
        colors=colors, markers=markers,
    )
    raw_dot_ci(
        axes[1], data, labels, "energy_per_offered_j", "Attributed task energy (J/offered request)",
        colors=colors, markers=markers,
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    save_figure(fig, output, "figS_load_scaling")
    return data


def breakdown_figure(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = frame[frame.method == "scd"].copy()
    data["probability"] = data.trace.map(
        lambda value: float(
            Path(value).name.split("_p", 1)[1].split("_seed", 1)[0].replace("em", "e-")
        )
    )
    data["strict_misses"] = data.admitted_requests - data.on_time_requests
    probabilities = sorted(data.probability.unique())
    labels = [f"{value:.0e}" for value in probabilities]
    data["plot_label"] = data.probability.map(
        {value: label for value, label in zip(probabilities, labels, strict=True)}
    )
    fig, ax = plt.subplots(figsize=(7.16, 4.6), constrained_layout=True)
    raw_dot_ci(
        ax,
        data,
        labels,
        "strict_misses",
        "Strict deadline misses in held-out interval",
        colors=[COLORS["scd"]] * len(labels),
        markers=[MARKERS["scd"]] * len(labels),
    )
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set_ylabel("Native thinning probability")
    ax.axvspan(-0.25, 0.25, color="#009E73", alpha=0.08, zorder=0)
    ax.text(
        0.01,
        0.02,
        "Green band: zero misses",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#006B4B",
    )
    save_figure(fig, output, "figS_breakdown")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--deadline", type=Path, required=True)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--breakdown", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    args.out.mkdir(parents=True, exist_ok=True)
    main_frame, main_protocol = load_campaign(args.campaign)
    network_frame, network_protocol = load_campaign(args.network)
    deadline_frame, deadline_protocol = load_campaign(args.deadline)
    load_frame, load_protocol = load_campaign(args.load)
    breakdown_frame, breakdown_protocol = load_campaign(args.breakdown)
    architecture_figure(args.out)
    primary = primary_figure(main_frame, args.out)
    sensitivity = sensitivity_figure(main_frame, args.out)
    effects = effects_figure(main_frame, network_frame, deadline_frame, args.out)
    load_source = load_scaling_figure(main_frame, load_frame, args.out)
    breakdown_source = breakdown_figure(breakdown_frame, args.out)
    primary.to_csv(args.out / "fig2_primary_source.csv", index=False)
    sensitivity.to_csv(args.out / "fig3_sampling_cache_source.csv", index=False)
    effects.to_csv(args.out / "fig4_effects_stress_source.csv", index=False)
    load_source.to_csv(args.out / "figS_load_scaling_source.csv", index=False)
    breakdown_source.to_csv(args.out / "figS_breakdown_source.csv", index=False)
    metadata = {
        "schema": "tnsm.strict-paper-figures.v1",
        "status": "PASS",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "campaigns": {
            "main": {
                "path": str(args.campaign.resolve()),
                "protocol_sha256": main_protocol["protocol_sha256"],
                "execution_source_sha256": main_protocol.get("execution_source_sha256", {}),
            },
            "network": {
                "path": str(args.network.resolve()),
                "protocol_sha256": network_protocol["protocol_sha256"],
                "execution_source_sha256": network_protocol.get("execution_source_sha256", {}),
            },
            "deadline": {
                "path": str(args.deadline.resolve()),
                "protocol_sha256": deadline_protocol["protocol_sha256"],
                "execution_source_sha256": deadline_protocol.get("execution_source_sha256", {}),
            },
            "load": {
                "path": str(args.load.resolve()),
                "protocol_sha256": load_protocol["protocol_sha256"],
                "execution_source_sha256": load_protocol.get("execution_source_sha256", {}),
            },
            "breakdown": {
                "path": str(args.breakdown.resolve()),
                "protocol_sha256": breakdown_protocol["protocol_sha256"],
                "execution_source_sha256": breakdown_protocol.get("execution_source_sha256", {}),
            },
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(args.out.iterdir())
            if path.is_file() and path.name != "figure_provenance.json"
        },
    }
    (args.out / "figure_provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "outputs": len(metadata["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
