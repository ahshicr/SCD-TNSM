"""Fail-closed consistency audit for the source-locked TNSM revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import fitz


CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent
MANUSCRIPT = PACKAGE / "manuscript"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_campaign(directory: Path) -> tuple[dict, list[dict], list[str]]:
    protocol_path = directory / "protocol.json"
    protocol = load_json(protocol_path)
    failures: list[str] = []
    run_paths = sorted((directory / "runs").glob("*.json"))
    expected = int(protocol.get("task_count", -1))
    if len(run_paths) != expected:
        failures.append(f"run count {len(run_paths)} != {expected}")
    records = [load_json(path) for path in run_paths]
    keys = [str(row.get("key")) for row in records]
    if len(keys) != len(set(keys)):
        failures.append("duplicate run key")

    source_hashes = protocol.get("execution_source_sha256", {})
    for filename, expected_hash in source_hashes.items():
        source = CODE / "src" / filename
        if not source.is_file():
            failures.append(f"missing execution source {filename}")
        elif digest(source) != expected_hash:
            failures.append(f"changed execution source {filename}")

    record_hash_fields = (
        "deployment_sha256",
        "simulator_sha256",
        "scheduler_core_sha256",
        "service_forecast_sha256",
    )
    for field in record_hash_fields:
        observed = {str(row.get(field)) for row in records}
        expected_value = protocol.get(field)
        if len(observed) != 1:
            failures.append(f"mixed {field}: {sorted(observed)}")
        elif expected_value is not None and observed != {str(expected_value)}:
            failures.append(f"record/protocol mismatch for {field}")

    required_integer_fields = (
        "evaluated_requests",
        "admitted_requests",
        "rejected_requests",
        "on_time_requests",
        "actual_cold_start_requests",
        "peak_aoi_samples",
    )
    for row in records:
        if int(row.get("admitted_requests", -1)) + int(row.get("rejected_requests", -1)) != int(
            row.get("evaluated_requests", -2)
        ):
            failures.append(f"outcome conservation failed for {row.get('key')}")
            break
        if any(not isinstance(row.get(field), int) for field in required_integer_fields):
            failures.append(f"integer provenance failed for {row.get('key')}")
            break
        for metric in (
            "success_ratio",
            "miss_ratio",
            "actual_cold_start_ratio",
            "jain_tenant_success",
        ):
            value = float(row.get(metric, -1.0))
            if not 0.0 <= value <= 1.0:
                failures.append(f"ratio {metric} outside [0,1] for {row.get('key')}")
                break
    return protocol, records, failures


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def pdf_tokens(path: Path) -> list[str]:
    with fitz.open(path) as document:
        text = " ".join(page.get_text() for page in document)
    return re.findall(r"[0-9A-Za-z]+", text)


def bibliography_keys(path: Path) -> set[str]:
    return set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", path.read_text(encoding="utf-8")))


def cited_keys(text: str) -> set[str]:
    found: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        found.update(value.strip() for value in group.split(",") if value.strip())
    return found


def tex_dependencies(roots: list[Path]) -> list[Path]:
    """Return only TeX sources reachable from the submitted root documents."""
    pending = [path.resolve() for path in roots]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        visited.add(path)
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            child = (path.parent / name)
            if child.suffix == "":
                child = child.with_suffix(".tex")
            if child.is_file():
                pending.append(child.resolve())
    return sorted(visited)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--deadline", type=Path, required=True)
    parser.add_argument("--breakdown", type=Path, required=True)
    parser.add_argument("--measured-startup", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--conformance", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--response-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    campaigns = {
        "main": (args.main, 618),
        "network": (args.network, 75),
        "load": (args.load, 30),
        "deadline": (args.deadline, 60),
        "breakdown": (args.breakdown, 165),
        "measured_startup": (args.measured_startup, 180),
    }
    protocols: dict[str, dict] = {}
    records: dict[str, list[dict]] = {}
    for name, (directory, expected) in campaigns.items():
        protocol, rows, failures = validate_campaign(directory)
        protocols[name] = protocol
        records[name] = rows
        check(
            f"{name} campaign is complete, conserved, and source locked",
            not failures and len(rows) == expected,
            {"runs": len(rows), "expected": expected, "failures": failures},
        )

    all_rows = [row for group in records.values() for row in group]
    check(
        "registered run total is exactly 1128 with unique keys",
        len(all_rows) == 1128 and len({row["key"] for row in all_rows}) == 1128,
        {"runs": len(all_rows), "unique_keys": len({row["key"] for row in all_rows})},
    )

    startup_protocol = protocols["measured_startup"]
    startup_rows = records["measured_startup"]
    startup_evidence = startup_protocol.get("startup_profile_evidence", {})
    registered_profile_hash = startup_protocol.get("startup_profile_evidence_sha256")
    evidence_without_hash = {
        name: value for name, value in startup_evidence.items() if name != "profile_sha256"
    }
    profile_counts = {
        name: sum(row.get("startup_profile") == name for row in startup_rows)
        for name in ("measured_low", "measured_central", "measured_high")
    }
    task_profile_hashes = {
        task["startup_profile"]: task["startup_profile_sha256"]
        for task in startup_protocol.get("tasks", [])
    }
    record_profile_failures = [
        row.get("key")
        for row in startup_rows
        if row.get("startup_profile_sha256")
        != task_profile_hashes.get(row.get("startup_profile"))
    ]
    check(
        "measured-startup profiles are complete and cryptographically bound to every run",
        profile_counts == {
            "measured_low": 60,
            "measured_central": 60,
            "measured_high": 60,
        }
        and startup_evidence.get("profile_sha256") == registered_profile_hash
        and canonical_hash(evidence_without_hash) == registered_profile_hash
        and not record_profile_failures,
        {
            "profile_counts": profile_counts,
            "profile_sha256": registered_profile_hash,
            "record_failures": record_profile_failures,
        },
    )

    paper_values_path = args.paper_output / "paper_values.json"
    paper_values = load_json(paper_values_path)
    expected_counts = {
        "primary_run_count": 618,
        "network_run_count": 75,
        "load_run_count": 30,
        "deadline_run_count": 60,
        "breakdown_run_count": 165,
        "measured_startup_run_count": 180,
        "total_registered_runs": 1128,
    }
    observed_counts = {name: int(paper_values.get(name, -1)) for name in expected_counts}
    check("paper value counts match all campaigns", observed_counts == expected_counts, observed_counts)

    all_registered_path = args.paper_output / "all_registered_runs.csv"
    check(
        "paper table source contains every registered run once",
        all_registered_path.is_file() and csv_row_count(all_registered_path) == 1128,
        str(all_registered_path.resolve()),
    )

    trace_registry_path = CODE / "results" / "strict" / "traces_v3" / "primary_trace_registry.json"
    trace_registry = load_json(trace_registry_path)
    trace_hash_failures = []
    for row in trace_registry.get("traces", []):
        manifest = CODE / str(row["path"]) / "manifest.json"
        if not manifest.is_file() or digest(manifest) != row["manifest_sha256"]:
            trace_hash_failures.append(str(row.get("directory")))
    check(
        "all 16 primary and sampling-control traces are registered and hash verified",
        trace_registry.get("status") == "PASS"
        and int(trace_registry.get("trace_count", 0)) == 16
        and not trace_hash_failures,
        {"traces": trace_registry.get("trace_count"), "failures": trace_hash_failures},
    )

    table_files = (
        "platform_results.tex",
        "nominal_results.tex",
        "sampling_results.tex",
        "load_scaling_results.tex",
        "breakdown_results.tex",
        "cache_capacity_results.tex",
        "component_ablation_results.tex",
        "energy_pressure_results.tex",
        "tenant_isolation_results.tex",
        "request_flood_results.tex",
        "power_uncertainty_results.tex",
        "network_stress_results.tex",
        "deadline_guard_results.tex",
        "exact_benchmark_results.tex",
        "measured_startup_table.tex",
        "paper_values_macros.tex",
    )
    missing_tables = [name for name in table_files if not (args.paper_output / name).is_file()]
    malformed_tables = []
    for name in table_files[:-1]:
        path = args.paper_output / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            if (
                not lines
                or "&" not in lines[0]
                or not lines[0].rstrip().endswith(r"\\")
                or lines[1:2] != [r"\midrule"]
                or r"\\midrule" in text
                or r"\\bottomrule" in text
                or "\t" in text
            ):
                malformed_tables.append(name)
    check(
        "all generated tables exist and use valid LaTeX control sequences",
        not missing_tables and not malformed_tables,
        {"missing": missing_tables, "malformed": malformed_tables},
    )

    table_provenance = load_json(args.paper_output / "table_provenance.json")
    table_hash_failures = []
    for name, expected_hash in table_provenance.get("outputs", {}).items():
        path = args.paper_output / name
        if not path.is_file() or digest(path) != expected_hash:
            table_hash_failures.append(name)
    check(
        "generated table provenance matches every output",
        table_provenance.get("status") == "PASS" and not table_hash_failures,
        table_hash_failures,
    )
    copied_table_failures = []
    for name, expected_hash in table_provenance.get("outputs", {}).items():
        if not name.endswith(".tex"):
            continue
        target = MANUSCRIPT / "sections" / name
        if not target.is_file() or digest(target) != expected_hash:
            copied_table_failures.append(name)
    check(
        "manuscript table fragments exactly match generated outputs",
        not copied_table_failures,
        copied_table_failures,
    )

    macro_text = (args.paper_output / "paper_values_macros.tex").read_text(
        encoding="utf-8"
    )
    macro_values = {
        name: float(value)
        for name, value in re.findall(
            r"\\newcommand\{\\([^}]+)\}\{([-+0-9.]+)\}", macro_text
        )
    }
    network_direction_failures = []
    for prefix in ("StressHundredPayloadFour", "StressTenPayloadSixteen"):
        expected_signs = {
            f"{prefix}IndependentMinusSCDSuccessPP": -1,
            f"{prefix}IndependentMinusSCDLatencyMs": 1,
            f"{prefix}IndependentMinusSCDQueueMs": 1,
        }
        for name, expected_sign in expected_signs.items():
            value = macro_values.get(name)
            if value is None or value * expected_sign <= 0.0:
                network_direction_failures.append({"macro": name, "value": value})
    check(
        "network-effect macros follow the declared challenger-minus-SCD direction",
        not network_direction_failures,
        network_direction_failures,
    )

    figure_provenance_path = args.figure_output / "figure_provenance.json"
    figure_provenance = load_json(figure_provenance_path)
    figure_hash_failures = []
    for name, expected_hash in figure_provenance.get("outputs", {}).items():
        path = args.figure_output / name
        if not path.is_file() or digest(path) != expected_hash:
            figure_hash_failures.append(name)
    check(
        "generated figure provenance matches every raster, vector, and source file",
        figure_provenance.get("status") == "PASS" and not figure_hash_failures,
        figure_hash_failures,
    )
    copied_figure_failures = []
    for name, expected_hash in figure_provenance.get("outputs", {}).items():
        target = MANUSCRIPT / "figures" / name
        if not target.is_file() or digest(target) != expected_hash:
            copied_figure_failures.append(name)
    check(
        "manuscript figures and source CSVs exactly match generated outputs",
        not copied_figure_failures,
        copied_figure_failures,
    )
    figure_source = (CODE / "src" / "strict_paper_figures.py").read_text(encoding="utf-8")
    check(
        "final figure generator contains no heat-map rendering path",
        "heatmap" not in figure_source.lower() and "imshow(" not in figure_source,
        "categorical raw points and interval summaries only",
    )

    exact = load_json(args.exact / "exact_benchmark_summary.json")
    check(
        "restricted exact benchmark is exhaustive and dominates greedy on all seeds",
        exact.get("status") == "PASS"
        and exact.get("exact_dominates_greedy_all_seeds") is True
        and int(exact.get("assignments_per_seed", 0)) * len(exact.get("seeds", [])) == 81920,
        {
            "assignments": int(exact.get("assignments_per_seed", 0)) * len(exact.get("seeds", [])),
            "mean_gap_percent": exact.get("mean_greedy_gap_percent"),
        },
    )

    conformance = load_json(args.conformance)
    check(
        "implementation conformance audit passes all executable and protocol checks",
        conformance.get("status") == "PASS"
        and int(conformance.get("failed", -1)) == 0
        and int(conformance.get("checks", 0)) >= 49,
        {
            "checks": conformance.get("checks"),
            "passed": conformance.get("passed"),
            "failed": conformance.get("failed"),
        },
    )

    suite = ET.parse(args.tests).getroot().find("testsuite")
    test_result = {
        name: int(suite.attrib[name]) for name in ("tests", "failures", "errors")
    }
    check(
        "complete unit suite passes",
        test_result["tests"] >= 128
        and test_result["failures"] == 0
        and test_result["errors"] == 0,
        test_result,
    )

    document_roots = [
        MANUSCRIPT / "main.tex",
        MANUSCRIPT / "main_clean.tex",
        MANUSCRIPT / "supplement.tex",
        MANUSCRIPT / "response.tex",
        MANUSCRIPT / "assess.tex",
    ]
    document_paths = tex_dependencies(document_roots)
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in document_paths)
    check(
        "manuscript, supplement, response, and assessment contain no semicolons",
        ";" not in source_text and "；" not in source_text,
        "ASCII and full-width forms checked",
    )
    stale_tokens = (
        "8,925",
        "1,405",
        "1174.49",
        "5.4254",
        "1.7202",
        "2.28\\%",
        "four-slot default",
        "trigger-derived tenants",
    )
    primary_paths = tex_dependencies(
        [MANUSCRIPT / "main.tex", MANUSCRIPT / "main_clean.tex", MANUSCRIPT / "supplement.tex"]
    )
    primary_text = "\n".join(path.read_text(encoding="utf-8") for path in primary_paths)
    present_stale = [token for token in stale_tokens if token in primary_text]
    check("no superseded aggregate-campaign claims remain", not present_stale, present_stale)

    response_files = {
        name: MANUSCRIPT / "responses" / f"{name}.tex"
        for name in ("associate_editor", "reviewer1", "reviewer2", "reviewer3")
    }
    response_texts = {
        name: path.read_text(encoding="utf-8") for name, path in response_files.items()
    }
    response_text = "\n".join(response_texts.values())
    retired_response_claims = (
        "44.7545",
        "25.0469",
        "46.0605",
        "14.4912",
        "4,055.25",
        "4,053.87",
        "4,054.12",
        "4,053.28",
        "approximately 8.872 MJ",
        "284.83",
        "258.21",
        "7.22\\pm9.12",
        "93.95\\%",
        "1.54\\%",
    )
    present_retired_response_claims = [
        token for token in retired_response_claims if token in response_text
    ]
    check(
        "review responses contain no retired grace-sweep, predictor, network, or instrumentation claims",
        not present_retired_response_claims,
        present_retired_response_claims,
    )
    synchronized_response_inputs = {
        "reviewer1 deadline evidence": (
            response_texts["reviewer1"], r"\input{sections/deadline_guard_results}"
        ),
        "reviewer3 deadline evidence": (
            response_texts["reviewer3"], r"\input{sections/deadline_guard_results}"
        ),
        "reviewer1 component evidence": (
            response_texts["reviewer1"], r"\input{sections/component_ablation_results}"
        ),
        "reviewer2 component evidence": (
            response_texts["reviewer2"], r"\input{sections/component_ablation_results}"
        ),
        "reviewer3 network evidence": (
            response_texts["reviewer3"], r"\input{sections/network_stress_results}"
        ),
    }
    missing_response_inputs = [
        name for name, (text, token) in synchronized_response_inputs.items() if token not in text
    ]
    check(
        "response evidence reuses the current generated deadline, component, and network tables",
        not missing_response_inputs,
        missing_response_inputs,
    )

    active_retired_figure_tokens = (
        "fig_main",
        "fig_stress",
        "fig_envelope",
        "fig_stress_source",
        "fig_envelope_source",
    )
    active_document_text = "\n".join(path.read_text(encoding="utf-8") for path in document_paths)
    present_retired_figure_tokens = [
        token for token in active_retired_figure_tokens if token in active_document_text
    ]
    check(
        "submitted TeX dependency graph contains no retired figure references",
        not present_retired_figure_tokens,
        present_retired_figure_tokens,
    )

    supplement_aux = (MANUSCRIPT / "supplement.aux").read_text(encoding="utf-8")
    supplementary_table_numbers = {
        int(value)
        for value in re.findall(r"\\newlabel\{tab:[^}]+\}\{\{S([0-9]+)\}", supplement_aux)
    }
    check(
        "supplementary table catalogue is contiguous from S1 through S20",
        supplementary_table_numbers == set(range(1, 21)),
        sorted(supplementary_table_numbers),
    )

    evaluation_text = (MANUSCRIPT / "sections" / "evaluation.tex").read_text(
        encoding="utf-8"
    )
    protocol_paragraph = evaluation_text.split("Every task has a unique scenario key", 1)[0]
    required_run_macros = {
        r"\StrictMainRuns",
        r"\StrictNetworkRuns",
        r"\StrictLoadRuns",
        r"\StrictDeadlineRuns",
        r"\StrictBreakdownRuns",
        r"\StrictMeasuredStartupRuns",
        r"\StrictTotalRuns",
    }
    missing_run_macros = sorted(required_run_macros - set(re.findall(r"\\[A-Za-z]+", protocol_paragraph)))
    check(
        "evaluation run-count paragraph enumerates every campaign before the total",
        not missing_run_macros,
        missing_run_macros,
    )

    bib_keys = bibliography_keys(MANUSCRIPT / "refs.bib")
    used_keys = cited_keys(source_text)
    check(
        "every cited key exists in the synchronized bibliography",
        used_keys <= bib_keys,
        sorted(used_keys - bib_keys),
    )
    recommended = {"Abedi2017Security", "Abedi2025Mobility", "Abedi2024MANO", "Akbari2021AoI"}
    check(
        "all four reviewer-recommended references are cited",
        recommended <= used_keys,
        sorted(recommended - used_keys),
    )
    check(
        "IEEE AI-assisted-content acknowledgment is present",
        "OpenAI Codex" in source_text and "take responsibility" in source_text,
        "current IEEE submission policy checked on 2026-09-14",
    )

    response_audit = load_json(args.response_audit)
    check(
        "every official comment is complete and every response starts with thanks",
        bool(response_audit.get("passed"))
        and int(response_audit.get("total", 0)) == 35,
        {"total": response_audit.get("total"), "passed": response_audit.get("passed")},
    )

    required_pdfs = ("main.pdf", "main_clean.pdf", "supplement.pdf", "response.pdf", "assess.pdf")
    missing_pdfs = [name for name in required_pdfs if not (MANUSCRIPT / name).is_file()]
    check("all five submission PDFs exist", not missing_pdfs, missing_pdfs)
    if not missing_pdfs:
        marked = pdf_tokens(MANUSCRIPT / "main.pdf")
        clean = pdf_tokens(MANUSCRIPT / "main_clean.pdf")
        check(
            "marked and clean manuscripts contain identical text",
            marked == clean,
            {"marked_tokens": len(marked), "clean_tokens": len(clean)},
        )
        with fitz.open(MANUSCRIPT / "main_clean.pdf") as document:
            page_count = len(document)
        check(
            "clean manuscript satisfies the 16-page submission ceiling",
            page_count <= 16,
            {"pages": page_count, "ceiling": 16},
        )

    report = {
        "schema": "tnsm.submission-consistency.v3",
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": len(checks),
        "passed": sum(bool(item["passed"]) for item in checks),
        "failed": sum(not bool(item["passed"]) for item in checks),
        "check_results": checks,
        "campaign_protocol_sha256": {
            name: digest(directory / "protocol.json")
            for name, (directory, _) in campaigns.items()
        },
        "document_sha256": {
            name: digest(MANUSCRIPT / name)
            for name in required_pdfs
            if (MANUSCRIPT / name).is_file()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checks", "passed", "failed")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
