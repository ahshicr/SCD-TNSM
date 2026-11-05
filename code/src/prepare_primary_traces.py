from __future__ import annotations

"""Build every native-resolution trace required by the main protocol."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

from src.strict_trace import build_request_trace, load_request_trace


SEEDS = tuple(range(20260914, 20260919))
PROBABILITY = 4.0e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for seed in SEEDS:
        specs.append(
            {
                "directory": f"binomial_p0004_seed{seed}",
                "seed": seed,
                "method": "binomial",
                "release_model": "uniform",
            }
        )
        specs.append(
            {
                "directory": f"unbiased_p0004_seed{seed}",
                "seed": seed,
                "method": "unbiased_round",
                "release_model": "uniform",
            }
        )
        specs.append(
            {
                "directory": f"binomial_stratified_p0004_seed{seed}",
                "seed": seed,
                "method": "binomial",
                "release_model": "stratified",
            }
        )
    specs.append(
        {
            "directory": "deterministic_p0004",
            "seed": SEEDS[0],
            "method": "deterministic_round",
            "release_model": "uniform",
        }
    )
    return specs


def _build(task: tuple[str, str, dict[str, object]]) -> dict[str, Any]:
    workload_text, root_text, spec = task
    output = Path(root_text) / str(spec["directory"])
    if output.exists():
        trace = load_request_trace(output, verify_hashes=True)
        manifest = trace.manifest
        sampling = manifest["sampling"]
        expected = {
            "probability": PROBABILITY,
            "seed": int(spec["seed"]),
            "method": str(spec["method"]),
            "release_model": str(spec["release_model"]),
        }
        actual = {name: sampling[name] for name in expected}
        if actual != expected:
            raise ValueError(f"existing trace has incompatible semantics: {output}")
    else:
        manifest = build_request_trace(
            Path(workload_text),
            output,
            probability=PROBABILITY,
            seed=int(spec["seed"]),
            method=str(spec["method"]),
            release_model=str(spec["release_model"]),
        )
    manifest_path = output / "manifest.json"
    return {
        **spec,
        "path": output.as_posix(),
        "requests": int(manifest["sampling"]["retained_requests"]),
        "expected_requests": float(manifest["sampling"]["expected_requests"]),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("results/strict/processed/workload_strict_v2.npz"),
    )
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not args.workload.is_file():
        raise FileNotFoundError(args.workload)
    args.trace_root.mkdir(parents=True, exist_ok=True)
    tasks = [(str(args.workload), str(args.trace_root), spec) for spec in trace_specs()]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: str(row["directory"]))
    registry = {
        "schema": "tnsm.primary-trace-registry.v1",
        "status": "PASS",
        "source": args.workload.as_posix(),
        "source_sha256": sha256(args.workload),
        "probability": PROBABILITY,
        "seeds": list(SEEDS),
        "trace_count": len(rows),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "traces": rows,
    }
    registry_path = args.trace_root / "primary_trace_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "traces": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
