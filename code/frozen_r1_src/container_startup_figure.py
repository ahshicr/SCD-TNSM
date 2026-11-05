from __future__ import annotations

"""Submission-grade visualization of the paired physical container protocol."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from src.container_trial_design import BLOCKS, PORTS, PULLS, design
from src.summarize_container_trials import audit_activation


BOARD_LABELS = {5: "Pi 5", 9: "Pi 4", 7: "Pi 3", 8: "Pi 2"}
METHODS = (("host_process", "Host process", "#0072B2", "o"),
           ("cached_container", "Cached container", "#D55E00", "s"))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.2, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.7, "axes.linewidth": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": .025,
    })


def collect(campaign: Path) -> tuple[list[dict], list[dict]]:
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    if not status.get("complete") or manifest.get("trials") != design():
        raise ValueError("The complete fixed physical protocol is required")
    activations: list[dict] = []
    acquisitions: list[dict] = []
    for port in PORTS:
        directory = campaign / f"port_{port:02d}"
        for spec in (item for item in manifest["trials"] if item["port"] == port):
            observation = json.loads((directory /
                f"activation_b{spec['block']:02d}_{spec['method']}.json").read_text(encoding="utf-8"))
            metrics = audit_activation(observation, spec)
            activations.append({"port": port, "board": BOARD_LABELS[port],
                                "block": spec["block"], "method": spec["method"], **metrics})
        for repetition in range(PULLS):
            observation = json.loads((directory / "runtime_evidence" /
                f"pull_{repetition:02d}_acquisition.json").read_text(encoding="utf-8"))
            result = observation["download"]
            acquisitions.append({"port": port, "board": BOARD_LABELS[port],
                                 "repetition": repetition,
                                 "success": int(result["returncode"] == 0),
                                 "elapsed_s": result["elapsed_s"]})
    if len(activations) != len(PORTS) * 2 * BLOCKS or len(acquisitions) != len(PORTS) * PULLS:
        raise ValueError("Missing physical observations")
    return activations, acquisitions


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))


def activation_values(rows: list[dict], port: int, method: str, metric: str) -> np.ndarray:
    values = np.asarray([row[metric] for row in rows
                         if row["port"] == port and row["method"] == method])
    if len(values) != BLOCKS:
        raise ValueError("A plotted activation cell is incomplete")
    return values


def launch_dumbbell_panel(ax, rows: list[dict], metric: str, xlabel: str) -> None:
    """Fill the log span with paired board-level host/container estimates."""
    y = np.arange(len(PORTS), dtype=float)
    for board_index, port in enumerate(PORTS):
        summaries = []
        for method_index, (method, label, color, marker) in enumerate(METHODS):
            values = activation_values(rows, port, method, metric)
            mean, half = mean_ci(values)
            summaries.append(mean)
            y_pos = y[board_index] + (-.10 if method_index == 0 else .10)
            ax.scatter(values, y_pos + np.linspace(-.035, .035, len(values)),
                       s=8, facecolors="none", edgecolors=color, marker=marker,
                       linewidths=.55, alpha=.60, zorder=3)
            ax.errorbar(mean, y_pos, xerr=half, color=color, fmt=marker,
                        markersize=3.8, capsize=2.0, linewidth=.9,
                        markerfacecolor="white", markeredgewidth=.65, zorder=5,
                        label=label if board_index == 0 else None)
        ax.plot(summaries, [y[board_index], y[board_index]], color="#A8A8A8",
                linewidth=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(y, [BOARD_LABELS[port] for port in PORTS])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linewidth=.35, color="#c8c8c8")
    ax.spines[["top", "right"]].set_visible(False)


def resident_deltas(rows: list[dict]) -> list[dict]:
    result = []
    for port in PORTS:
        selected = [row for row in rows if row["port"] == port]
        by_block = {block: {} for block in range(BLOCKS)}
        for row in selected:
            by_block[row["block"]][row["method"]] = row["resident_response_ms"]
        for block, values in by_block.items():
            if set(values) != {"host_process", "cached_container"}:
                raise ValueError("Resident-response pairing is incomplete")
            result.append({"port": port, "board": BOARD_LABELS[port], "block": block,
                           "cached_minus_host_ms":
                           values["cached_container"] - values["host_process"]})
    return result


def resident_delta_panel(ax, rows: list[dict]) -> None:
    x = np.arange(len(PORTS), dtype=float)
    for index, port in enumerate(PORTS):
        values = np.asarray([row["cached_minus_host_ms"] for row in rows
                             if row["port"] == port])
        mean, half = mean_ci(values)
        color = "#7A5195" if mean >= 0 else "#2F7F6F"
        ax.bar(x[index], mean, width=.58, color=color, alpha=.65,
               edgecolor=color, linewidth=.65, hatch="///", zorder=2)
        ax.scatter(x[index] + np.linspace(-.10, .10, len(values)), values,
                   s=8, facecolors="white", edgecolors=color,
                   linewidths=.5, alpha=.70, zorder=4)
        ax.errorbar(x[index], mean, yerr=half, fmt="none", color="#303030",
                    capsize=2.0, linewidth=.8, zorder=5)
    ax.axhline(0, color="#555555", linestyle=":", linewidth=.75)
    ax.set_xticks(x, [BOARD_LABELS[port] for port in PORTS])
    ax.set_ylabel("Cached minus host\nresident response (ms)")
    ax.grid(axis="y", linewidth=.35, color="#c8c8c8")
    ax.spines[["top", "right"]].set_visible(False)


def acquisition_panel(ax, rows: list[dict]) -> None:
    x = np.arange(len(PORTS), dtype=float)
    for index, port in enumerate(PORTS):
        values = np.asarray([row["elapsed_s"] for row in rows if row["port"] == port])
        success = [row["success"] for row in rows if row["port"] == port]
        if len(values) != PULLS or not all(success):
            raise ValueError("All five recorded acquisition attempts must be plotted and successful")
        mean, half = mean_ci(values)
        ax.bar(x[index], mean, width=.58, color="#009E73", alpha=.62,
               edgecolor="#007A5A", linewidth=.65, hatch="\\\\", zorder=2)
        ax.scatter(x[index] + np.linspace(-.10, .10, len(values)), values,
                   s=12, facecolors="white", edgecolors="#007A5A",
                   marker="D", linewidths=.65, zorder=4)
        ax.errorbar(x[index], mean, yerr=half, fmt="none", color="#303030",
                    capsize=2.0, linewidth=.8, zorder=5)
    ax.set_xticks(x, [BOARD_LABELS[port] for port in PORTS])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Empty-store acquisition (s)")
    ax.grid(axis="y", linewidth=.35, color="#c8c8c8")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activations, acquisitions = collect(args.campaign)
    deltas = resident_deltas(activations)
    write_csv(activations, args.output_dir / "fig_container_startup_source_activations.csv")
    write_csv(acquisitions, args.output_dir / "fig_container_startup_source_acquisitions.csv")
    write_csv(deltas, args.output_dir / "fig_container_startup_source_resident_deltas.csv")

    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.05), constrained_layout=True)
    launch_dumbbell_panel(axes[0, 0], activations, "ready_ms", "Launch to ready (ms)")
    launch_dumbbell_panel(axes[0, 1], activations, "first_prediction_ms",
                          "Launch to first result (ms)")
    resident_delta_panel(axes[1, 0], deltas)
    acquisition_panel(axes[1, 1], acquisitions)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside upper center",
               ncol=2, columnspacing=1.2, handletextpad=.35,
               borderaxespad=.05)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-.13, 1.03, f"({label})", transform=ax.transAxes,
                fontweight="bold", va="bottom", ha="left")
    base = args.output_dir / "fig_container_startup"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(args.output_dir / "fig_container_startup_preview.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"activation_points": len(activations),
                      "acquisition_points": len(acquisitions),
                      "output": str(base)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
