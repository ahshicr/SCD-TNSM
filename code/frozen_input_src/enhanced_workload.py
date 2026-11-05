from __future__ import annotations

"""Build the compact, auditable workload used by the revision experiments.

The recovered artifact stores the public Azure Functions trace at function/minute
resolution.  The simulator does not materialize billions of individual requests;
instead, it preserves their counts and first moments in 16 duration--memory
classes and four service tenants for every one of the 20,160 trace minutes.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


TENANT_NAMES = np.asarray(
    ["interactive", "event-driven", "periodic", "orchestration"], dtype="U32"
)


def _tenant_lookup(trigger_names: np.ndarray) -> np.ndarray:
    mapping = {
        "http": 0,
        "event": 1,
        "queue": 1,
        "storage": 1,
        "timer": 2,
        "orchestration": 3,
        "others": 3,
        "unknown": 3,
    }
    return np.asarray([mapping.get(str(name).lower(), 3) for name in trigger_names], dtype=np.int16)


def build_aggregates(
    workload_path: Path,
    out_path: Path,
    duration_bins: int = 4,
    memory_bins: int = 4,
) -> dict[str, object]:
    data = np.load(workload_path, allow_pickle=True)
    counts = data["counts"]
    p50_ms = data["p50_ms"]
    p99_ms = data["p99_ms"]
    memory_mb = data["memory_mb"]
    trigger_ids = data["trigger_ids"]
    trigger_names = data["trigger_names"]

    n_functions, minutes = counts.shape
    days = p50_ms.shape[1]
    med_duration = np.median(p50_ms, axis=1)
    duration_edges = np.quantile(
        med_duration, np.linspace(0.0, 1.0, duration_bins + 1)[1:-1]
    )
    memory_edges = np.quantile(
        memory_mb, np.linspace(0.0, 1.0, memory_bins + 1)[1:-1]
    )
    duration_class = np.searchsorted(duration_edges, med_duration, side="right")
    memory_class = np.searchsorted(memory_edges, memory_mb, side="right")
    class_ids = (duration_class * memory_bins + memory_class).astype(np.int16)
    tenant_by_trigger = _tenant_lookup(trigger_names)
    tenant_ids = tenant_by_trigger[trigger_ids]

    classes = duration_bins * memory_bins
    tenants = len(TENANT_NAMES)
    shape = (minutes, classes, tenants)
    agg_count = np.zeros(shape, dtype=np.float64)
    agg_compute = np.zeros(shape, dtype=np.float64)
    agg_tail = np.zeros(shape, dtype=np.float64)
    agg_payload = np.zeros(shape, dtype=np.float64)

    for f in range(n_functions):
        cls = int(class_ids[f])
        tenant = int(tenant_ids[f])
        payload_per_invocation = max(0.015, math.sqrt(float(memory_mb[f])) * 0.018)
        for day in range(days):
            start = day * 1440
            end = min(minutes, start + 1440)
            if start >= minutes:
                break
            values = counts[f, start:end].astype(np.float64)
            if not np.any(values):
                continue
            agg_count[start:end, cls, tenant] += values
            agg_compute[start:end, cls, tenant] += values * float(p50_ms[f, day])
            agg_tail[start:end, cls, tenant] += values * float(p99_ms[f, day])
            agg_payload[start:end, cls, tenant] += values * payload_per_invocation

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        count=agg_count,
        compute=agg_compute,
        tail=agg_tail,
        payload=agg_payload,
        tenant_names=TENANT_NAMES,
        duration_edges=duration_edges,
        memory_edges=memory_edges,
    )
    summary: dict[str, object] = {
        "source": str(workload_path.as_posix()),
        "functions": int(n_functions),
        "minutes": int(minutes),
        "classes": int(classes),
        "tenants": int(tenants),
        "total_invocations": int(round(float(agg_count.sum()))),
        "active_class_tenant_minutes": int(np.count_nonzero(agg_count)),
        "tenant_invocations": {
            str(TENANT_NAMES[t]): int(round(float(agg_count[:, :, t].sum())))
            for t in range(tenants)
        },
        "duration_edges_ms": [float(x) for x in duration_edges],
        "memory_edges_mb": [float(x) for x in memory_edges],
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="results/processed/workload.npz")
    parser.add_argument(
        "--out", default="results/revision/workload_aggregate_16x4.npz"
    )
    args = parser.parse_args()
    summary = build_aggregates(Path(args.workload), Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
