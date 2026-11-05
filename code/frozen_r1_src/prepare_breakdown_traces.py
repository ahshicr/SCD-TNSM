from __future__ import annotations

"""Build native-resolution traces for the reviewer-requested breakdown search."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

from src.strict_trace import build_request_trace


PROBABILITIES = (
    1.0e-7,
    2.0e-7,
    5.0e-7,
    1.0e-6,
    2.0e-6,
    5.0e-6,
    1.0e-5,
    2.0e-5,
    5.0e-5,
    1.0e-4,
    2.0e-4,
)
SEEDS = tuple(range(20260914, 20260919))


def probability_label(probability: float) -> str:
    return f"{probability:.0e}".replace("+", "").replace("-", "m")


def trace_directory(root: Path, probability: float, seed: int) -> Path:
    return root / f"breakdown_p{probability_label(probability)}_seed{seed}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(task: tuple[str, str, float, int]) -> dict[str, Any]:
    workload_text, output_text, probability, seed = task
    output = Path(output_text)
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["sampling"]["method"] != "binomial"
            or float(manifest["sampling"]["probability"]) != probability
            or int(manifest["sampling"]["seed"]) != seed
            or manifest["sampling"]["release_model"] != "uniform"
        ):
            raise ValueError(f"existing trace has incompatible semantics: {output}")
    else:
        manifest = build_request_trace(
            Path(workload_text),
            output,
            probability=probability,
            seed=seed,
            method="binomial",
            release_model="uniform",
        )
    manifest_path = output / "manifest.json"
    return {
        "directory": output.as_posix(),
        "probability": probability,
        "seed": seed,
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
    tasks = [
        (
            str(args.workload),
            str(trace_directory(args.trace_root, probability, seed)),
            probability,
            seed,
        )
        for probability in PROBABILITIES
        for seed in SEEDS
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: (row["probability"], row["seed"]))
    registry = {
        "schema": "tnsm.breakdown-trace-registry.v1",
        "source": args.workload.as_posix(),
        "source_sha256": sha256(args.workload),
        "probabilities": list(PROBABILITIES),
        "seeds": list(SEEDS),
        "sampling": "native function-minute binomial thinning",
        "release_model": "uniform within source minute",
        "traces": rows,
    }
    registry_path = args.trace_root / "breakdown_trace_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "traces": len(rows), "registry": str(registry_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
