from __future__ import annotations

"""Build and load request-level traces for the strict simulator.

The Azure source records invocation counts for each anonymised function in each
minute.  The production trace builder applies binomial thinning at that native
function-minute resolution.  Conditional on a retained invocation, its release
time is uniform within the source minute, which is the standard Poisson-process
interpretation when only interval counts are observed.  Every scheduler in a
paired experiment consumes the same immutable trace and the same per-request
random variates.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


TRACE_SCHEMA = "tnsm.strict-request-trace.v3"
NETWORK_RANDOM_DIM = 4
TENANT_NAMES = np.asarray(
    ["logical-tenant-0", "logical-tenant-1", "logical-tenant-2", "logical-tenant-3"],
    dtype="U32",
)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _tenant_from_application(app_ids: np.ndarray) -> np.ndarray:
    """Create four reproducible, application-coherent logical tenants.

    The public trace exposes anonymised application identities but not cloud
    account owners.  Grouping by the low two bits of a stable SHA-256 digest
    keeps every function from an application in one logical tenant without
    inspecting future demand or confusing trigger type with tenancy.
    """

    return (app_ids.astype(np.uint64, copy=False) % np.uint64(4)).astype(np.int8)


def stable_anonymous_id(value: str) -> np.int64:
    """Return a process-independent anonymised 63-bit identifier."""

    raw = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return np.int64(int.from_bytes(raw, "big") & ((1 << 63) - 1))


def _classify_functions(
    p50_ms: np.ndarray,
    memory_mb: np.ndarray,
    bins: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median_duration = np.median(p50_ms, axis=1)
    duration_edges = np.quantile(
        median_duration, np.linspace(0.0, 1.0, bins + 1)[1:-1]
    )
    memory_edges = np.quantile(
        memory_mb, np.linspace(0.0, 1.0, bins + 1)[1:-1]
    )
    duration_class = np.searchsorted(duration_edges, median_duration, side="right")
    memory_class = np.searchsorted(memory_edges, memory_mb, side="right")
    strata = (duration_class * bins + memory_class).astype(np.int8)
    return strata, duration_edges.astype(np.float32), memory_edges.astype(np.float32)


def _sample_sparse_cells(
    counts: np.ndarray,
    probability: float,
    rng: np.random.Generator,
    method: str,
    function_block: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    function_parts: list[np.ndarray] = []
    minute_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    total = 0
    for start in range(0, counts.shape[0], function_block):
        stop = min(counts.shape[0], start + function_block)
        source = counts[start:stop]
        if method == "binomial":
            sampled = rng.binomial(source, probability).astype(np.int32)
        elif method == "unbiased_round":
            scaled = source.astype(np.float64) * probability
            base = np.floor(scaled).astype(np.int32)
            sampled = base + (rng.random(source.shape) < (scaled - base))
            sampled = sampled.astype(np.int32)
        elif method == "deterministic_round":
            sampled = np.rint(source.astype(np.float64) * probability).astype(np.int32)
        else:
            raise ValueError(f"unknown thinning method: {method}")
        local_function, minute = np.nonzero(sampled)
        retained = sampled[local_function, minute]
        function_parts.append((local_function + start).astype(np.int32))
        minute_parts.append(minute.astype(np.int32))
        count_parts.append(retained.astype(np.int32))
        total += int(retained.sum(dtype=np.int64))
    return function_parts, minute_parts, count_parts, total


def _save_array(directory: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = directory / f"{name}.npy"
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_file(path),
    }


def build_request_trace(
    workload_path: Path,
    output_directory: Path,
    *,
    probability: float = 4.0e-4,
    seed: int = 20260914,
    method: str = "binomial",
    release_model: str = "uniform",
    function_block: int = 128,
) -> dict[str, object]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing trace directory: {output_directory}"
        )
    workload = np.load(workload_path, allow_pickle=False)
    required = {
        "counts",
        "p50_ms",
        "p99_ms",
        "memory_mb",
        "trigger_ids",
        "trigger_names",
        "function_keys",
        "app_ids",
        "duration_quantiles_ms",
        "duration_quantile_probabilities",
        "duration_imputed",
    }
    missing = required.difference(workload.files)
    if missing:
        raise KeyError(f"processed workload is missing arrays: {sorted(missing)}")

    counts = workload["counts"]
    p50_ms = workload["p50_ms"].astype(np.float32, copy=False)
    p99_ms = workload["p99_ms"].astype(np.float32, copy=False)
    memory_mb = workload["memory_mb"].astype(np.float32, copy=False)
    if counts.ndim != 2 or counts.shape[0] != p50_ms.shape[0]:
        raise ValueError("counts and function metadata have inconsistent dimensions")
    if counts.shape[1] != p50_ms.shape[1] * 1440:
        raise ValueError("duration metadata must contain one column per trace day")
    if np.any(counts < 0) or not np.all(np.isfinite(p50_ms)):
        raise ValueError("workload contains negative counts or non-finite durations")
    if np.any(p99_ms < p50_ms) or np.any(memory_mb <= 0):
        raise ValueError("invalid tail duration or memory metadata")

    if release_model not in {"uniform", "stratified", "minute_start"}:
        raise ValueError(f"unknown within-minute release model: {release_model}")
    streams = np.random.SeedSequence(seed).spawn(5)
    sampling_rng = np.random.Generator(np.random.PCG64(streams[0]))
    release_rng = np.random.Generator(np.random.PCG64(streams[1]))
    service_rng = np.random.Generator(np.random.PCG64(streams[2]))
    network_rng = np.random.Generator(np.random.PCG64(streams[3]))
    startup_rng = np.random.Generator(np.random.PCG64(streams[4]))
    function_parts, minute_parts, count_parts, event_count = _sample_sparse_cells(
        counts, probability, sampling_rng, method, function_block
    )
    release_s = np.empty(event_count, dtype=np.float64)
    function_id = np.empty(event_count, dtype=np.int32)
    event_id = np.arange(event_count, dtype=np.int64)
    service_z = np.empty(event_count, dtype=np.float32)
    network_z = np.empty((event_count, NETWORK_RANDOM_DIM), dtype=np.float32)
    startup_z = np.empty(event_count, dtype=np.float32)

    cursor = 0
    retained_cells = 0
    for functions, minutes, retained in zip(
        function_parts, minute_parts, count_parts, strict=True
    ):
        expanded_functions = np.repeat(functions, retained)
        expanded_minutes = np.repeat(minutes, retained)
        size = len(expanded_functions)
        if size == 0:
            continue
        slc = slice(cursor, cursor + size)
        function_id[slc] = expanded_functions
        release_u = release_rng.random(size)
        if release_model == "uniform":
            offset_s = release_u * 60.0
        elif release_model == "stratified":
            offset_s = np.empty(size, dtype=np.float64)
            local_cursor = 0
            for retained_count in retained:
                cell_stop = local_cursor + int(retained_count)
                offset_s[local_cursor:cell_stop] = (
                    np.arange(int(retained_count), dtype=np.float64)
                    + release_u[local_cursor:cell_stop]
                ) * (60.0 / int(retained_count))
                local_cursor = cell_stop
        else:
            offset_s = np.zeros(size, dtype=np.float64)
        release_s[slc] = expanded_minutes.astype(np.float64) * 60.0 + offset_s
        service_z[slc] = service_rng.standard_normal(size).astype(np.float32)
        network_z[slc] = network_rng.standard_normal(
            (size, NETWORK_RANDOM_DIM)
        ).astype(np.float32)
        startup_z[slc] = startup_rng.standard_normal(size).astype(np.float32)
        cursor += size
        retained_cells += len(functions)
    if cursor != event_count:
        raise AssertionError(f"trace fill mismatch: {cursor} != {event_count}")

    order = np.lexsort((event_id, release_s))
    release_s = release_s[order]
    function_id = function_id[order]
    event_id = event_id[order]
    service_z = service_z[order]
    network_z = network_z[order]
    startup_z = startup_z[order]
    if np.any(np.diff(release_s) < 0):
        raise AssertionError("release times are not monotone")

    app_ids = workload["app_ids"].astype(np.int64, copy=False)
    tenant_id = _tenant_from_application(app_ids)
    strata, duration_edges, memory_edges = _classify_functions(p50_ms, memory_mb)
    function_key_digest = np.asarray(
        [stable_anonymous_id(str(value)) for value in workload["function_keys"]],
        dtype=np.int64,
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        arrays: dict[str, dict[str, object]] = {}
        for name, value in {
            "release_s": release_s,
            "function_id": function_id,
            "event_id": event_id,
            "service_z": service_z,
            "network_z": network_z,
            "startup_z": startup_z,
            "p50_ms": p50_ms,
            "p99_ms": p99_ms,
            "duration_quantiles_ms": workload["duration_quantiles_ms"].astype(
                np.float32, copy=False
            ),
            "duration_quantile_probabilities": workload[
                "duration_quantile_probabilities"
            ].astype(np.float64, copy=False),
            "memory_mb": memory_mb,
            "tenant_id": tenant_id,
            "stratum_id": strata,
            "app_id": app_ids,
            "trigger_id": workload["trigger_ids"].astype(np.int16, copy=False),
            "duration_imputed": workload["duration_imputed"].astype(
                np.uint8, copy=False
            ),
            "function_key_digest": function_key_digest,
        }.items():
            arrays[name] = _save_array(temporary, name, value)

        source_total = int(counts.sum(dtype=np.int64))
        expected = float(source_total) * probability
        variance = float(source_total) * probability * (1.0 - probability)
        z_score = (event_count - expected) / math.sqrt(max(variance, 1.0))
        manifest: dict[str, object] = {
            "schema": TRACE_SCHEMA,
            "source": {
                "path": workload_path.as_posix(),
                "sha256": sha256_file(workload_path),
                "functions": int(counts.shape[0]),
                "minutes": int(counts.shape[1]),
                "invocations": source_total,
                "nonzero_function_minutes": int(np.count_nonzero(counts)),
            },
            "sampling": {
                "method": method,
                "probability": probability,
                "seed": seed,
                "bit_generator": "PCG64",
                "independent_random_streams": [
                    "thinning",
                    "release",
                    "service",
                    "network",
                    "startup",
                ],
                "function_block": function_block,
                "release_model": release_model,
                "retained_requests": event_count,
                "retained_function_minutes": retained_cells,
                "expected_requests": expected,
                "binomial_standardized_deviation": z_score,
            },
            "semantics": {
                "cache_identity": "anonymised Azure function",
                "release_within_minute": release_model,
                "service_randomness": "shared standard-normal variate per request",
                "network_randomness": "shared standard-normal variate per request",
                "startup_randomness": "shared standard-normal variate per request",
                "stratum_use": "reporting and SLA configuration only",
                "tenant_assignment": (
                    "four synthetic logical tenants from stable anonymised-application "
                    "SHA-256 modulo 4; trigger type is not treated as tenant identity"
                ),
            },
            "tenant_names": TENANT_NAMES.tolist(),
            "duration_edges_ms": duration_edges.tolist(),
            "memory_edges_mb": memory_edges.tolist(),
            "arrays": arrays,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, output_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


@dataclass(frozen=True)
class RequestTrace:
    directory: Path
    manifest: dict[str, object]
    release_s: np.ndarray
    function_id: np.ndarray
    event_id: np.ndarray
    service_z: np.ndarray
    network_z: np.ndarray
    startup_z: np.ndarray
    p50_ms: np.ndarray
    p99_ms: np.ndarray
    duration_quantiles_ms: np.ndarray
    duration_quantile_probabilities: np.ndarray
    memory_mb: np.ndarray
    tenant_id: np.ndarray
    stratum_id: np.ndarray
    app_id: np.ndarray
    trigger_id: np.ndarray
    duration_imputed: np.ndarray
    function_key_digest: np.ndarray

    @property
    def requests(self) -> int:
        return int(self.release_s.shape[0])


def load_request_trace(directory: Path, *, verify_hashes: bool = False) -> RequestTrace:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unsupported trace schema: {manifest.get('schema')}")
    arrays: dict[str, np.ndarray] = {}
    for name, record in manifest["arrays"].items():
        path = directory / record["file"]
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"hash mismatch for {path}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"shape or dtype mismatch for {path}")
        arrays[name] = value
    request_length = len(arrays["release_s"])
    for name in ("function_id", "event_id", "service_z", "network_z", "startup_z"):
        if len(arrays[name]) != request_length:
            raise ValueError(f"request array length mismatch: {name}")
    if np.any(np.diff(arrays["release_s"]) < 0):
        raise ValueError("release times are not sorted")
    function_length = len(arrays["memory_mb"])
    for name in (
        "tenant_id",
        "stratum_id",
        "app_id",
        "trigger_id",
        "function_key_digest",
    ):
        if len(arrays[name]) != function_length:
            raise ValueError(f"function array length mismatch: {name}")
    quantiles = arrays["duration_quantiles_ms"]
    probabilities = arrays["duration_quantile_probabilities"]
    if quantiles.ndim != 3 or quantiles.shape[2] != len(probabilities):
        raise ValueError("duration quantile arrays have incompatible dimensions")
    if np.any(np.diff(probabilities) <= 0) or probabilities[0] != 0 or probabilities[-1] != 1:
        raise ValueError("duration quantile probabilities must increase from zero to one")
    if np.any(np.diff(quantiles, axis=2) < 0):
        raise ValueError("duration quantile values are not monotone")
    if arrays["duration_imputed"].shape != quantiles.shape[:2]:
        raise ValueError("duration-imputation mask has incompatible dimensions")
    return RequestTrace(directory=directory, manifest=manifest, **arrays)


def iter_request_blocks(trace: RequestTrace, block_size: int) -> Iterator[slice]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for start in range(0, trace.requests, block_size):
        yield slice(start, min(trace.requests, start + block_size))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=Path("results/processed/workload.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probability", type=float, default=4.0e-4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--method",
        choices=("binomial", "unbiased_round", "deterministic_round"),
        default="binomial",
    )
    parser.add_argument(
        "--release-model",
        choices=("uniform", "stratified", "minute_start"),
        default="uniform",
    )
    parser.add_argument("--function-block", type=int, default=128)
    args = parser.parse_args()
    manifest = build_request_trace(
        args.workload,
        args.out,
        probability=args.probability,
        seed=args.seed,
        method=args.method,
        release_model=args.release_model,
        function_block=args.function_block,
    )
    print(json.dumps(manifest["sampling"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
