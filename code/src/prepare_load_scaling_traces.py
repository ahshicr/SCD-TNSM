from __future__ import annotations

"""Build source-locked request traces for the registered load sweep."""

import argparse
import hashlib
import json
from pathlib import Path

from src.strict_trace import build_request_trace, load_request_trace


LOADS = (
    {"label": "load_0p5", "probability": 2.0e-4, "directory": "binomial_p0002"},
    {"label": "load_2", "probability": 8.0e-4, "directory": "binomial_p0008"},
)
SEEDS = tuple(range(20260914, 20260919))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("results/strict/processed/workload_strict_v2.npz"),
    )
    parser.add_argument(
        "--trace-root", type=Path, default=Path("results/strict/traces_v3")
    )
    args = parser.parse_args()
    args.trace_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for load in LOADS:
        for seed in SEEDS:
            directory = args.trace_root / f"{load['directory']}_seed{seed}"
            if directory.exists():
                trace = load_request_trace(directory, verify_hashes=True)
                sampling = trace.manifest["sampling"]
                if (
                    sampling["method"] != "binomial"
                    or sampling["release_model"] != "uniform"
                    or int(sampling["seed"]) != seed
                    or float(sampling["probability"]) != float(load["probability"])
                ):
                    raise ValueError(f"existing trace has a different protocol: {directory}")
                manifest = trace.manifest
            else:
                manifest = build_request_trace(
                    args.workload,
                    directory,
                    probability=float(load["probability"]),
                    seed=seed,
                    method="binomial",
                    release_model="uniform",
                )
            manifest_path = directory / "manifest.json"
            records.append(
                {
                    "label": load["label"],
                    "probability": load["probability"],
                    "seed": seed,
                    "directory": directory.as_posix(),
                    "requests": int(manifest["sampling"]["retained_requests"]),
                    "manifest_sha256": sha256(manifest_path),
                }
            )
            print(json.dumps(records[-1]), flush=True)
    registry = {
        "schema": "tnsm.load-scaling-traces.v1",
        "workload": args.workload.as_posix(),
        "workload_sha256": sha256(args.workload),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "traces": records,
    }
    registry_path = args.trace_root / "load_scaling_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "traces": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
