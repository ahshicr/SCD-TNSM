from __future__ import annotations

"""Run the genuine exhaustive finite-horizon benchmark on five trace seeds."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.strict_config import load_deployment
from src.strict_optimal_benchmark import run_benchmark
from src.strict_trace import load_request_trace


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_suite(
    trace_root: Path,
    deployment_path: Path,
    output: Path,
    jobs: int,
    ports: tuple[int, ...],
) -> dict[str, object]:
    if jobs < 1 or jobs > 10:
        raise ValueError("jobs must be between 1 and 10")
    deployment = load_deployment(deployment_path)
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for seed in range(20260914, 20260919):
        trace_path = trace_root / f"binomial_p0004_seed{seed}"
        trace = load_request_trace(trace_path, verify_hashes=True)
        result = run_benchmark(trace, deployment, jobs=jobs, ports=ports)
        rows.append({"seed": seed, **asdict(result)})
        manifest_path = trace_path / "manifest.json"
        traces.append(
            {
                "seed": seed,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
            }
        )
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "exact_benchmark_runs.csv", index=False)
    report = {
        "schema": "tnsm.exact-small-instance-suite.v1",
        "status": "PASS",
        "seeds": list(range(20260914, 20260919)),
        "jobs": jobs,
        "ports": list(ports),
        "assignments_per_seed": int(frame["enumerated_assignments"].iloc[0]),
        "exact_dominates_greedy_all_seeds": bool(
            (frame["exact_objective"] <= frame["greedy_objective"] + 1.0e-12).all()
        ),
        "mean_greedy_gap_percent": float(frame["greedy_gap_percent"].mean()),
        "minimum_greedy_gap_percent": float(frame["greedy_gap_percent"].min()),
        "maximum_greedy_gap_percent": float(frame["greedy_gap_percent"].max()),
        "deployment": str(deployment_path.resolve()),
        "deployment_sha256": sha256(deployment_path),
        "benchmark_source_sha256": sha256(Path(__file__).with_name("strict_optimal_benchmark.py")),
        "runner_source_sha256": sha256(Path(__file__)),
        "traces": traces,
        "interpretation": (
            "The exhaustive solver replays the full finite-state transitions on deliberately small instances. "
            "It validates the optimization implementation but is not a claim that the trace-scale online policy is globally optimal."
        ),
    }
    if not report["exact_dominates_greedy_all_seeds"]:
        report["status"] = "FAIL"
    (output / "exact_benchmark_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument(
        "--deployment", type=Path, default=Path("config/strict_deployment_20260914.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--ports", default="1,9,23")
    args = parser.parse_args()
    report = run_suite(
        args.trace_root,
        args.deployment,
        args.out,
        args.jobs,
        tuple(int(item) for item in args.ports.split(",")),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
