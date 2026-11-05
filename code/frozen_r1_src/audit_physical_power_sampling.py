"""Audit the temporal support of physical-campaign PoE measurements.

The audit is intentionally read-only with respect to the campaign.  It uses the
same midpoint timestamps and support rule as ``summarize_physical_trials`` and
separates gaps inside a measurement interval from planned authentication gaps
that begin after its final observation.  No samples are imputed and the 2.5 s
in-window continuity threshold is not relaxed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


POE_PORTS = (5, 6, 7, 8, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_power(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    durations: list[float] = []
    sessions: list[int] = []
    samples: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            start = float(row["query_start"])
            end = float(row["query_end"])
            observed = {
                int(port["port"]): port["poe_output_w"] for port in row["ports"]
            }
            missing = [port for port in POE_PORTS if observed.get(port) is None]
            if missing:
                raise ValueError(
                    f"Missing PoE readings for ports {missing} on power.jsonl line {line_number}"
                )
            times.append((start + end) / 2.0)
            durations.append(end - start)
            sessions.append(int(row["session"]))
            samples.append(int(row["sample"]))
    return (
        np.asarray(times, dtype=float),
        np.asarray(durations, dtype=float),
        np.asarray(sessions, dtype=int),
        np.asarray(samples, dtype=int),
    )


def interval_support(times: np.ndarray, start: float, stop: float) -> dict:
    if stop <= start:
        raise ValueError("Non-positive audit interval")
    if times.size < 2 or times[0] > start:
        before_start = int(np.searchsorted(times, start)) - 1
        before_stop = int(np.searchsorted(times, stop)) - 1
        return {
            "supported": False,
            "reason": "missing_boundary_support",
            "interval_start": start,
            "interval_stop": stop,
            "point_count": int(np.count_nonzero((times >= start) & (times <= stop))),
            "max_gap_s": None,
            "start_distance_to_previous_sample_s": (
                float(start - times[before_start]) if before_start >= 0 else None
            ),
            "stop_distance_from_previous_sample_s": (
                float(stop - times[before_stop]) if before_stop >= 0 else None
            ),
            "stop_distance_to_next_sample_s": (
                float(times[before_stop + 1] - stop)
                if before_stop + 1 < len(times)
                else None
            ),
            "offending_gaps": [],
            "terminal_hold_required": None,
            "two_sided_bracketing_would_reject": True,
        }

    first = max(0, int(np.searchsorted(times, start)) - 1)
    last = min(len(times), int(np.searchsorted(times, stop)) + 1)
    selected = times[first:last]
    diffs = np.diff(selected)
    offending: list[dict] = []
    for local_index in np.flatnonzero(diffs > 2.5):
        global_index = first + int(local_index)
        offending.append(
            {
                "left_sample_index": global_index,
                "right_sample_index": global_index + 1,
                "left_time": float(times[global_index]),
                "right_time": float(times[global_index + 1]),
                "gap_s": float(times[global_index + 1] - times[global_index]),
                "overlap_s": float(
                    max(
                        0.0,
                        min(stop, times[global_index + 1])
                        - max(start, times[global_index]),
                    )
                ),
                "location": (
                    "crosses_start"
                    if times[global_index] < start
                    else "crosses_stop"
                    if times[global_index + 1] > stop
                    else "internal"
                ),
            }
        )
    before_start = int(np.searchsorted(times, start)) - 1
    before_stop = int(np.searchsorted(times, stop)) - 1
    start_gap = float(times[right_start] - times[before_start]) if (
        (right_start := int(np.searchsorted(times, start, side="left"))) < len(times)
        and (before_start := int(np.searchsorted(times, start, side="right")) - 1) >= 0
        and times[right_start] != start
    ) else 0.0
    before_stop = int(np.searchsorted(times, stop, side="right")) - 1
    after_stop = int(np.searchsorted(times, stop, side="left"))
    tail = float(stop - times[before_stop]) if before_stop >= 0 else float("inf")
    stop_bracket_gap = (
        float(times[after_stop] - times[before_stop])
        if before_stop >= 0 and after_stop < len(times) and times[after_stop] != stop
        else 0.0
    )
    internal_offending = [gap for gap in offending if gap["location"] != "crosses_stop"]
    supported = bool(start_gap <= 2.5 and tail <= 2.5 and not internal_offending)
    terminal_hold = bool(after_stop >= len(times) or stop_bracket_gap > 2.5)
    two_sided_reject = bool(after_stop >= len(times) or offending)
    return {
        "supported": supported,
        "reason": None if supported else "sampling_gap_above_2.5_s_inside_interval",
        "interval_start": start,
        "interval_stop": stop,
        "point_count": int(np.count_nonzero((times >= start) & (times <= stop))),
        "max_gap_s": float(np.max(diffs)) if diffs.size else None,
        "start_distance_to_previous_sample_s": float(start - times[before_start]),
        "stop_distance_from_previous_sample_s": float(stop - times[before_stop]),
        "stop_distance_to_next_sample_s": (
            float(times[before_stop + 1] - stop)
            if before_stop + 1 < len(times)
            else None
        ),
        "offending_gaps": offending,
        "in_window_offending_gaps": internal_offending,
        "terminal_hold_required": terminal_hold,
        "two_sided_bracketing_would_reject": two_sided_reject,
    }


def percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p50", "p95", "p99", "max"),
            np.percentile(array, (0, 50, 95, 99, 100)),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    power_path = campaign / "power.jsonl"
    times, durations, sessions, samples = load_power(power_path)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Power midpoint timestamps are not strictly increasing")
    if not np.array_equal(samples, np.arange(samples.size)):
        raise ValueError("Power sample numbers are not contiguous")

    gaps = np.diff(times)
    global_gap_rows: list[dict] = []
    for index in np.flatnonzero(gaps > 2.5):
        i = int(index)
        global_gap_rows.append(
            {
                "left_sample_index": i,
                "right_sample_index": i + 1,
                "gap_s": float(gaps[i]),
                "left_session": int(sessions[i]),
                "right_session": int(sessions[i + 1]),
                "crosses_session_boundary": bool(sessions[i] != sessions[i + 1]),
            }
        )

    trial_rows: list[dict] = []
    for trial_dir in sorted(campaign.glob("trial_*")):
        summary_path = trial_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protocol = summary["protocol"]
        row = {
            "ordinal": int(protocol["ordinal"]),
            "block": int(protocol["block"]),
            "cohort": protocol["cohort"],
            "scenario": protocol["scenario"],
            "method": protocol["method"],
        }
        if protocol["cohort"] == "poe":
            arrival = float(summary["arrival_start_monotonic"])
            measurement_end = float(summary["measurement_end_monotonic"])
            row["idle_baseline"] = interval_support(times, arrival - 15.0, arrival)
            row["measurement"] = interval_support(times, arrival, measurement_end)
            row["energy_supported"] = bool(
                row["idle_baseline"]["supported"] and row["measurement"]["supported"]
            )
        else:
            row["idle_baseline"] = None
            row["measurement"] = None
            row["energy_supported"] = None
        trial_rows.append(row)

    poe_rows = [row for row in trial_rows if row["cohort"] == "poe"]
    affected = [row for row in poe_rows if not row["energy_supported"]]
    two_sided_rejections = [
        row for row in poe_rows
        if row["idle_baseline"]["two_sided_bracketing_would_reject"]
        or row["measurement"]["two_sided_bracketing_would_reject"]
    ]
    terminal_holds = [
        row for row in poe_rows if row["measurement"]["terminal_hold_required"]
    ]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "campaign_complete": json.loads(
            (campaign / "campaign_status.json").read_text(encoding="utf-8")
        )["complete"],
        "power_jsonl_sha256": sha256(power_path),
        "analysis_source_sha256": sha256(Path(__file__)),
        "sampling_threshold_s": 2.5,
        "power_sample_count": int(times.size),
        "query_duration_s": percentiles(durations),
        "inter_sample_gap_s": percentiles(gaps),
        "global_gaps_above_threshold": global_gap_rows,
        "global_gap_count": len(global_gap_rows),
        "global_gaps_crossing_session_boundary": sum(
            row["crosses_session_boundary"] for row in global_gap_rows
        ),
        "trial_count": len(trial_rows),
        "poe_trial_count": len(poe_rows),
        "poe_trials_with_complete_energy_support": len(poe_rows) - len(affected),
        "poe_trials_affected": len(affected),
        "affected_ordinals": [row["ordinal"] for row in affected],
        "two_sided_bracketing_rejection_count": len(two_sided_rejections),
        "two_sided_bracketing_rejection_ordinals": [
            row["ordinal"] for row in two_sided_rejections
        ],
        "terminal_hold_count": len(terminal_holds),
        "terminal_hold_ordinals": [row["ordinal"] for row in terminal_holds],
        "maximum_terminal_hold_s": max(
            row["measurement"]["stop_distance_from_previous_sample_s"]
            for row in terminal_holds
        ),
        "trials": trial_rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "power_sample_count": report["power_sample_count"],
                "global_gap_count": report["global_gap_count"],
                "session_boundary_gaps": report["global_gaps_crossing_session_boundary"],
                "poe_trial_count": report["poe_trial_count"],
                "poe_trials_affected": report["poe_trials_affected"],
                "affected_ordinals": report["affected_ordinals"],
                "two_sided_bracketing_rejections": report[
                    "two_sided_bracketing_rejection_count"
                ],
                "terminal_hold_count": report["terminal_hold_count"],
                "maximum_terminal_hold_s": report["maximum_terminal_hold_s"],
                "max_gap_s": report["inter_sample_gap_s"]["max"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
