from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def extract_archive(archive: Path, out_dir: Path) -> None:
    marker = out_dir / ".complete"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def day_index(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit(".d", 1)[1]) - 1


def collect_function_totals(files: list[Path]) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    totals: dict[str, int] = defaultdict(int)
    triggers: dict[str, str] = {}
    apps: dict[str, str] = {}
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            minute_cols = range(4, len(header))
            for row in reader:
                key = row[2]
                total = 0
                for idx in minute_cols:
                    value = row[idx]
                    if value:
                        total += int(float(value))
                totals[key] += total
                triggers.setdefault(key, row[3])
                apps.setdefault(key, row[1])
    return totals, triggers, apps


def load_counts(files: list[Path], chosen: list[str]) -> np.ndarray:
    index = {key: i for i, key in enumerate(chosen)}
    days = max(day_index(p) for p in files) + 1
    counts = np.zeros((len(chosen), days * 1440), dtype=np.int32)
    for path in files:
        day = day_index(path)
        offset = day * 1440
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            for row in reader:
                pos = index.get(row[2])
                if pos is None:
                    continue
                values = [int(float(x)) if x else 0 for x in row[4:]]
                counts[pos, offset : offset + len(values)] = values
    return counts


def load_duration(files: list[Path], chosen: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {key: i for i, key in enumerate(chosen)}
    days = max(day_index(p) for p in files) + 1
    p50 = np.ones((len(chosen), days), dtype=np.float32)
    p99 = np.ones((len(chosen), days), dtype=np.float32)
    seen = np.zeros((len(chosen), days), dtype=bool)
    for path in files:
        day = day_index(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pos = index.get(row["HashFunction"])
                if pos is None:
                    continue
                p50[pos, day] = max(float(row.get("percentile_Average_50") or 1.0), 1.0)
                p99[pos, day] = max(float(row.get("percentile_Average_99") or 1.0), p50[pos, day])
                seen[pos, day] = True
    for i in range(len(chosen)):
        valid = seen[i]
        med = float(np.median(p50[i, valid])) if valid.any() else 10.0
        tail = float(np.median(p99[i, valid])) if valid.any() else max(20.0, med)
        p50[i, ~valid] = med
        p99[i, ~valid] = tail
    return p50, p99


def load_memory(files: list[Path], apps: dict[str, str], chosen: list[str]) -> np.ndarray:
    wanted_apps = {apps[k] for k in chosen if k in apps}
    app_memory: dict[str, list[float]] = defaultdict(list)
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                app = row["HashApp"]
                if app not in wanted_apps:
                    continue
                value = row.get("AverageAllocatedMb") or row.get("AverageAllocatedMb_pct50") or "128"
                app_memory[app].append(max(float(value), 1.0))
    result = np.full(len(chosen), 128.0, dtype=np.float32)
    for i, key in enumerate(chosen):
        vals = app_memory.get(apps.get(key, ""))
        if vals:
            result[i] = float(np.median(vals))
    return result


def write_summary(out_dir: Path, counts: np.ndarray, chosen: list[str], triggers: dict[str, str]) -> None:
    total_invocations = int(counts.sum())
    active_minutes = int((counts.sum(axis=0) > 0).sum())
    trigger_counts: dict[str, int] = defaultdict(int)
    for key in chosen:
        trigger_counts[triggers.get(key, "unknown")] += 1
    summary = {
        "functions": len(chosen),
        "minutes": int(counts.shape[1]),
        "active_minutes": active_minutes,
        "total_invocations": total_invocations,
        "trigger_counts": dict(sorted(trigger_counts.items())),
    }
    (out_dir / "workload_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--out-dir", default="results/processed")
    parser.add_argument("--functions", type=int, default=9000)
    args = parser.parse_args()

    archive = Path(args.archive)
    work_dir = Path(args.work_dir) / "azure2019"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, work_dir)

    invocation_files = sorted(work_dir.glob("invocations_per_function_md.anon.d*.csv"))
    duration_files = sorted(work_dir.glob("function_durations_percentiles.anon.d*.csv"))
    memory_files = sorted(work_dir.glob("app_memory_percentiles.anon.d*.csv"))
    totals, triggers, apps = collect_function_totals(invocation_files)
    chosen = [k for k, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[: args.functions]]
    counts = load_counts(invocation_files, chosen)
    p50, p99 = load_duration(duration_files, chosen)
    memory = load_memory(memory_files, apps, chosen)

    trigger_names = sorted(set(triggers.get(k, "unknown") for k in chosen))
    trigger_index = {name: i for i, name in enumerate(trigger_names)}
    trigger_ids = np.array([trigger_index[triggers.get(k, "unknown")] for k in chosen], dtype=np.int16)
    # Python's built-in hash is salted for every interpreter process.  Using it
    # here made a regenerated workload differ even when the source archive and
    # command line were identical.  The first 63 bits of SHA-256 are stable,
    # deterministic and sufficient for an anonymised grouping identifier.
    app_ids = np.array(
        [
            int.from_bytes(
                hashlib.sha256(apps.get(k, "").encode("utf-8")).digest()[:8],
                "big",
            )
            & ((1 << 63) - 1)
            for k in chosen
        ],
        dtype=np.int64,
    )

    np.savez_compressed(
        out_dir / "workload.npz",
        counts=counts,
        p50_ms=p50,
        p99_ms=p99,
        memory_mb=memory,
        trigger_ids=trigger_ids,
        app_ids=app_ids,
        function_keys=np.array(chosen),
        trigger_names=np.array(trigger_names),
    )
    write_summary(out_dir, counts, chosen, triggers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
