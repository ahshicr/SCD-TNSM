from __future__ import annotations

"""Augment the processed Azure workload with all published duration quantiles."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


QUANTILE_COLUMNS = (
    "percentile_Average_0",
    "percentile_Average_1",
    "percentile_Average_25",
    "percentile_Average_50",
    "percentile_Average_75",
    "percentile_Average_99",
    "percentile_Average_100",
)
QUANTILE_PROBABILITIES = np.asarray([0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def stable_id(value: str) -> np.int64:
    digest = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return np.int64(int.from_bytes(digest, "big") & ((1 << 63) - 1))


def day_number(name: str) -> int:
    return int(name.rsplit(".d", 1)[1].split(".", 1)[0]) - 1


def build_strict_workload(
    workload_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, object]:
    source = np.load(workload_path, allow_pickle=False)
    function_keys = source["function_keys"]
    function_index = {str(key): index for index, key in enumerate(function_keys)}
    functions = len(function_keys)
    days = source["p50_ms"].shape[1]
    quantiles = np.full((functions, days, len(QUANTILE_COLUMNS)), np.nan, dtype=np.float32)
    app_hash = np.full(functions, "", dtype="U64")
    seen = np.zeros((functions, days), dtype=bool)

    with tarfile.open(archive_path, "r:xz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.name.startswith("function_durations_percentiles.anon.d")
            ),
            key=lambda member: member.name,
        )
        if len(members) != days:
            raise ValueError(f"expected {days} duration files, found {len(members)}")
        for member in members:
            day = day_number(member.name)
            binary = archive.extractfile(member)
            if binary is None:
                raise OSError(f"could not read {member.name}")
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                for row in csv.DictReader(text):
                    index = function_index.get(row["HashFunction"])
                    if index is None:
                        continue
                    values = []
                    for column in QUANTILE_COLUMNS:
                        raw = row.get(column)
                        values.append(max(float(raw) if raw else 1.0, 1.0))
                    quantiles[index, day] = np.maximum.accumulate(
                        np.asarray(values, dtype=np.float32)
                    )
                    app_hash[index] = row["HashApp"]
                    seen[index, day] = True

    missing_functions = 0
    for function in range(functions):
        valid = seen[function]
        if valid.any():
            replacement = np.median(quantiles[function, valid], axis=0)
        else:
            missing_functions += 1
            replacement = np.asarray([1, 1, 1, 10, 10, 20, 20], dtype=np.float32)
        quantiles[function, ~valid] = replacement
        quantiles[function] = np.maximum.accumulate(quantiles[function], axis=1)

    extracted_p50 = quantiles[:, :, 3]
    extracted_p99 = quantiles[:, :, 5]
    if not np.array_equal(extracted_p50, source["p50_ms"]):
        difference = float(np.max(np.abs(extracted_p50 - source["p50_ms"])))
        raise AssertionError(f"extracted p50 differs from processed source by {difference}")
    if not np.array_equal(extracted_p99, source["p99_ms"]):
        difference = float(np.max(np.abs(extracted_p99 - source["p99_ms"])))
        raise AssertionError(f"extracted p99 differs from processed source by {difference}")
    if np.any(np.diff(quantiles, axis=2) < 0):
        raise AssertionError("duration quantiles are not monotone")
    if np.any(app_hash == ""):
        # A missing app label is kept stable and unique to the function instead
        # of reverting to Python's process-salted hash.
        app_hash[app_hash == ""] = function_keys[app_hash == ""]
    stable_app_id = np.asarray([stable_id(value) for value in app_hash], dtype=np.int64)
    duration_imputed = (~seen).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: source[name] for name in source.files if name != "app_ids"}
    arrays.update(
        {
            "app_ids": stable_app_id,
            "duration_quantiles_ms": quantiles,
            "duration_quantile_probabilities": QUANTILE_PROBABILITIES,
            "duration_imputed": duration_imputed,
        }
    )
    np.savez_compressed(output_path, **arrays)
    report: dict[str, object] = {
        "schema": "tnsm.strict-workload.v1",
        "source_workload": workload_path.as_posix(),
        "source_workload_sha256": sha256_file(workload_path),
        "source_archive": archive_path.as_posix(),
        "source_archive_sha256": sha256_file(archive_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "functions": functions,
        "days": days,
        "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
        "missing_function_duration_rows": missing_functions,
        "imputed_function_days": int(duration_imputed.sum()),
        "p50_exact_match": True,
        "p99_exact_match": True,
        "stable_application_hash": "SHA-256 first 63 bits",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=Path("results/processed/workload.npz"))
    parser.add_argument("--archive", type=Path, default=Path("data/azurefunctions_dataset2019.tar.xz"))
    parser.add_argument("--out", type=Path, default=Path("results/strict/processed/workload_strict.npz"))
    args = parser.parse_args()
    report = build_strict_workload(args.workload, args.archive, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
