from __future__ import annotations

"""Compare two complete strict campaigns field by field.

The comparison separates expected provenance changes from scientific output
changes. It is used when refactoring inactive parameters so a no-op cleanup
cannot silently alter any metric or assignment count.
"""

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TO_DIFFER = {
    "key",
    "scenario_parameters",
    "deployment_sha256",
    "simulator_sha256",
    "scheduler_core_sha256",
    "runtime_s",
}


def semantic_identity(record: dict[str, Any], ignored_scenario_fields: set[str]) -> str:
    scenario = {
        key: value
        for key, value in record["scenario_parameters"].items()
        if key not in ignored_scenario_fields
    }
    identity = {
        "family": record["family"],
        "trace": Path(record["trace"]).name,
        "method": record["method"],
        "scenario": scenario,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def load_runs(
    directory: Path, ignored_scenario_fields: set[str]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((directory / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = semantic_identity(record, ignored_scenario_fields)
        if key in rows:
            raise AssertionError(f"duplicate key {key}")
        rows[key] = record
    if not rows:
        raise FileNotFoundError(f"no run files in {directory / 'runs'}")
    return rows


def compare(
    left: Path,
    right: Path,
    allowed_scientific_fields: set[str] | None = None,
    ignored_scenario_fields: set[str] | None = None,
) -> dict[str, Any]:
    allowed_scientific_fields = allowed_scientific_fields or set()
    ignored_scenario_fields = ignored_scenario_fields or set()
    a = load_runs(left, ignored_scenario_fields)
    b = load_runs(right, ignored_scenario_fields)
    missing_left = sorted(set(b) - set(a))
    missing_right = sorted(set(a) - set(b))
    differences: list[dict[str, Any]] = []
    provenance_differences: list[dict[str, Any]] = []
    allowed_scientific_differences: list[dict[str, Any]] = []
    for key in sorted(set(a) & set(b)):
        fields = sorted(set(a[key]) | set(b[key]))
        for field in fields:
            if a[key].get(field) == b[key].get(field):
                continue
            item = {
                "key": key,
                "field": field,
                "left": a[key].get(field),
                "right": b[key].get(field),
            }
            if field in EXPECTED_TO_DIFFER:
                provenance_differences.append(item)
            elif field in allowed_scientific_fields:
                allowed_scientific_differences.append(item)
            else:
                differences.append(item)
    return {
        "schema": "tnsm.strict-campaign-comparison.v1",
        "status": "PASS" if not missing_left and not missing_right and not differences else "FAIL",
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "left_runs": len(a),
        "right_runs": len(b),
        "missing_from_left": missing_left,
        "missing_from_right": missing_right,
        "ignored_scenario_fields": sorted(ignored_scenario_fields),
        "scientific_differences": differences,
        "allowed_scientific_fields": sorted(allowed_scientific_fields),
        "allowed_scientific_differences": allowed_scientific_differences,
        "expected_provenance_differences": provenance_differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-field", action="append", default=[])
    parser.add_argument("--ignore-scenario-field", action="append", default=[])
    args = parser.parse_args()
    report = compare(
        args.left,
        args.right,
        set(args.allow_field),
        set(args.ignore_scenario_field),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "left_runs": report["left_runs"],
                "right_runs": report["right_runs"],
                "scientific_differences": len(report["scientific_differences"]),
                "allowed_scientific_differences": len(report["allowed_scientific_differences"]),
                "expected_provenance_differences": len(report["expected_provenance_differences"]),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
