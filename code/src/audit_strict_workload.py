from __future__ import annotations

"""Full integrity and statistical audit of the strict workload and traces."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.strict_trace import TRACE_SCHEMA, _tenant_from_application, load_request_trace


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload-report",
        type=Path,
        default=ROOT / "results" / "strict" / "processed" / "workload_strict_v2.json",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=ROOT / "results" / "strict" / "traces_v3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "strict" / "workload_audit_20260914.json",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "details": details})

    workload_report = json.loads(args.workload_report.read_text(encoding="utf-8"))
    workload_path = ROOT / workload_report["output"]
    source_workload = ROOT / workload_report["source_workload"]
    source_archive = ROOT / workload_report["source_archive"]
    check("strict workload hash", sha256(workload_path) == workload_report["output_sha256"],
          workload_report["output_sha256"])
    check("processed source hash", sha256(source_workload) == workload_report["source_workload_sha256"],
          workload_report["source_workload_sha256"])
    check("Azure archive hash", sha256(source_archive) == workload_report["source_archive_sha256"],
          workload_report["source_archive_sha256"])
    with np.load(workload_path, allow_pickle=False) as workload:
        counts = workload["counts"]
        total_invocations = int(counts.sum(dtype=np.int64))
        nonzero_cells = int(np.count_nonzero(counts))
        check("9000 functions", counts.shape[0] == 9000, list(counts.shape))
        check("14 days at minute resolution", counts.shape[1] == 14 * 1440, list(counts.shape))
        check("source invocation total", total_invocations == 12_444_637_039,
              total_invocations)
        check("duration quantiles monotone",
              bool(np.all(np.diff(workload["duration_quantiles_ms"], axis=2) >= 0)),
              list(workload["duration_quantiles_ms"].shape))
        check("p50 preserved exactly",
              bool(np.array_equal(workload["duration_quantiles_ms"][:, :, 3], workload["p50_ms"])),
              None)
        check("p99 preserved exactly",
              bool(np.array_equal(workload["duration_quantiles_ms"][:, :, 5], workload["p99_ms"])),
              None)
        check("imputation count recorded",
              int(workload["duration_imputed"].sum()) == workload_report["imputed_function_days"],
              int(workload["duration_imputed"].sum()))

    expected_names = (
        [f"binomial_p0004_seed{seed}" for seed in range(20260914, 20260919)]
        + [f"unbiased_p0004_seed{seed}" for seed in range(20260914, 20260919)]
        + ["deterministic_p0004"]
        + [f"binomial_stratified_p0004_seed{seed}" for seed in range(20260914, 20260919)]
    )
    actual_names = sorted(path.name for path in args.trace_root.iterdir() if path.is_dir())
    check("complete preregistered trace set", sorted(expected_names) == actual_names, actual_names)

    manifests: dict[str, dict[str, Any]] = {}
    count_rows: list[dict[str, Any]] = []
    shared_metadata_hash: dict[str, str] | None = None
    metadata_names = (
        "p50_ms", "p99_ms", "duration_quantiles_ms",
        "duration_quantile_probabilities", "memory_mb", "tenant_id",
        "stratum_id", "app_id", "trigger_id", "duration_imputed",
        "function_key_digest",
    )
    for name in expected_names:
        directory = args.trace_root / name
        trace = load_request_trace(directory, verify_hashes=True)
        manifest = trace.manifest
        manifests[name] = manifest
        check(f"{name}: schema", manifest["schema"] == TRACE_SCHEMA, manifest["schema"])
        check(f"{name}: strict workload hash",
              manifest["source"]["sha256"] == workload_report["output_sha256"],
              manifest["source"]["sha256"])
        check(f"{name}: monotone release times",
              bool(np.all(np.diff(trace.release_s) >= 0)), trace.requests)
        check(f"{name}: request array lengths",
              len(trace.function_id) == len(trace.event_id) == len(trace.service_z)
              == len(trace.startup_z) == len(trace.network_z) == trace.requests,
              trace.requests)
        check(f"{name}: function index range",
              int(trace.function_id.min()) >= 0 and int(trace.function_id.max()) < 9000,
              [int(trace.function_id.min()), int(trace.function_id.max())])
        check(f"{name}: release offsets valid",
              bool(np.all((trace.release_s % 60.0) >= 0.0)
                   and np.all((trace.release_s % 60.0) < 60.0)), None)
        check(f"{name}: application-coherent tenants",
              bool(np.array_equal(trace.tenant_id, _tenant_from_application(trace.app_id))), None)
        current_metadata = {
            key: manifest["arrays"][key]["sha256"] for key in metadata_names
        }
        if shared_metadata_hash is None:
            shared_metadata_hash = current_metadata
        check(f"{name}: immutable function metadata",
              current_metadata == shared_metadata_hash, current_metadata)
        sampling = manifest["sampling"]
        count_rows.append(
            {
                "trace": name,
                "method": sampling["method"],
                "release_model": sampling["release_model"],
                "seed": sampling["seed"],
                "requests": sampling["retained_requests"],
                "expected": sampling["expected_requests"],
                "standardized_deviation": sampling["binomial_standardized_deviation"],
            }
        )

    for seed in range(20260914, 20260919):
        uniform = manifests[f"binomial_p0004_seed{seed}"]["arrays"]
        stratified = manifests[f"binomial_stratified_p0004_seed{seed}"]["arrays"]
        uniform_dir = args.trace_root / f"binomial_p0004_seed{seed}"
        stratified_dir = args.trace_root / f"binomial_stratified_p0004_seed{seed}"
        uniform_event = np.load(uniform_dir / "event_id.npy", mmap_mode="r")
        stratified_event = np.load(stratified_dir / "event_id.npy", mmap_mode="r")
        uniform_order = np.argsort(uniform_event)
        stratified_order = np.argsort(stratified_event)
        paired_names = ("function_id", "service_z", "network_z", "startup_z")
        paired_equal = len(uniform_event) == len(stratified_event)
        for key in paired_names:
            left = np.load(uniform_dir / f"{key}.npy", mmap_mode="r")
            right = np.load(stratified_dir / f"{key}.npy", mmap_mode="r")
            paired_equal = paired_equal and bool(
                np.array_equal(left[uniform_order], right[stratified_order])
            )
        check(
            f"seed {seed}: uniform/stratified paired sampling and randomness",
            paired_equal,
            list(paired_names),
        )
        check(
            f"seed {seed}: release reconstruction differs",
            uniform["release_s"]["sha256"] != stratified["release_s"]["sha256"],
            None,
        )

    binomial_z = [
        abs(row["standardized_deviation"])
        for row in count_rows
        if row["method"] == "binomial" and row["release_model"] == "uniform"
    ]
    check("binomial totals within four standard deviations", max(binomial_z) < 4.0,
          binomial_z)
    deterministic = next(row for row in count_rows if row["method"] == "deterministic_round")
    deterministic_loss = deterministic["requests"] - deterministic["expected"]
    check("deterministic rounding bias is retained and quantified",
          deterministic_loss < -0.10 * deterministic["expected"], deterministic_loss)

    failed = [row for row in checks if row["status"] == "FAIL"]
    report = {
        "schema": "tnsm.strict-workload-audit.v1",
        "workload_report": str(args.workload_report),
        "workload_report_sha256": sha256(args.workload_report),
        "trace_root": str(args.trace_root),
        "source_invocations": total_invocations,
        "nonzero_function_minutes": nonzero_cells,
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
        "trace_counts": count_rows,
        "failed_checks": failed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "schema", "source_invocations", "nonzero_function_minutes",
        "checks", "passed", "failed", "status"
    )}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
