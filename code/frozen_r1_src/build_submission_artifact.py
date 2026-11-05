from __future__ import annotations

"""Build a credential-free, machine-readable first-revision artifact.

Selection is an explicit allowlist.  LAN inventories, switch transcripts,
remote command logs, credentials, private keys and high-rate internal event
logs are never traversed by this builder.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


PACKAGE = Path(__file__).resolve().parents[2]
PROJECT = PACKAGE.parent
PRIVATE_IPV4 = re.compile(
    r"(?<![0-9.])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|"
    r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9.])"
)
MAC_ADDRESS = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def add_files(selected: set[Path], root: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        selected.update(path for path in root.glob(pattern) if path.is_file())


def selection() -> list[Path]:
    code = PACKAGE / "code"
    manuscript = PACKAGE / "manuscript"
    selected: set[Path] = set()
    for name in ("README.md", "requirements.txt", "pytest.ini", "run_revision.py",
                 "run_all.py", "IMPLEMENTATION_SCOPE.md", "IMPLEMENTATION_TRACEABILITY.md",
                 "IMPLEMENTATION_AUDIT.md",
                 "CONTAINER_EXPERIMENT_PROTOCOL.md"):
        selected.add(code / name)
    source_names = (
        "__init__.py",
        "build_strict_workload.py",
        "prepare_primary_traces.py",
        "prepare_load_scaling_traces.py",
        "prepare_breakdown_traces.py",
        "run_strict_campaign.py",
        "run_network_stress_extension.py",
        "run_load_scaling_extension.py",
        "run_deadline_guard_extension.py",
        "run_breakdown_extension.py",
        "run_measured_startup_strict_extension.py",
        "startup_profiles.py",
        "summarize_strict_campaign.py",
        "strict_optimal_benchmark.py",
        "strict_components.py",
        "run_exact_benchmark_suite.py",
        "strict_paper_tables.py",
        "strict_paper_figures.py",
        "strict_simulate.py",
        "strict_trace.py",
        "strict_config.py",
        "scheduler_core.py",
        "audit_implementation_conformance.py",
        "audit_submission_consistency.py",
        "audit_review_responses.py",
        "audit_strict_workload.py",
        "audit_strict_configuration.py",
        "compare_strict_campaigns.py",
        "theory_certificate.py",
        "thermal_dynamics.py",
        "forecast_online.py",
        "forecast_v2.py",
        "board_capabilities.py",
        "board_inference_client.py",
        "board_inference_worker.py",
        "board_pipeline.py",
        "board_calibration.py",
        "portable_inference.py",
        "calibrate_inference_boards.py",
        "summarize_inference_calibration.py",
        "calibrate_activation_transport.py",
        "summarize_activation_transport.py",
        "calibrate_power_thermal.py",
        "summarize_power_thermal.py",
        "fit_thermal_calibration.py",
        "poe_sampling_audit.py",
        "switch_transport.py",
        "physical_policy.py",
        "physical_trial_design.py",
        "run_physical_trials.py",
        "summarize_physical_trials.py",
        "physical_trace_audit.py",
        "audit_physical_power_sampling.py",
        "container_trial_design.py",
        "container_runtime_remote.py",
        "run_container_trials.py",
        "summarize_container_trials.py",
        "archive_container_campaign.py",
        "archive_physical_campaign.py",
        "export_portable_model.py",
        "physical_calibration_figure.py",
        "physical_scheduler_figure.py",
        "container_startup_figure.py",
        "validate_power_sessions.py",
        "build_revision_documents.py",
        "render_document_review.py",
        "build_submission_artifact.py",
    )
    selected.update(code / "src" / name for name in source_names)
    test_names = (
        "test_board_inference.py",
        "test_board_pipeline.py",
        "test_breakdown_protocol.py",
        "test_conserved_work.py",
        "test_container_trials.py",
        "test_hardness_construction.py",
        "test_load_scaling_protocol.py",
        "test_physical_analysis.py",
        "test_physical_policy.py",
        "test_physical_trial_design.py",
        "test_power_sampler_sessions.py",
        "test_power_thermal_accounting.py",
        "test_prequential_forecast.py",
        "test_primary_trace_protocol.py",
        "test_revision_artifact.py",
        "test_scheduler_core.py",
        "test_strict_campaign_protocol.py",
        "test_strict_components.py",
        "test_strict_optimal_benchmark.py",
        "test_strict_simulator.py",
        "test_switch_transport.py",
        "test_theory_certificate.py",
        "test_thermal_dynamics.py",
    )
    selected.update(code / "tests" / name for name in test_names)
    selected.add(code / "data" / "azurefunctions_dataset2019.tar.xz")
    selected.add(code / "results" / "strict" / "processed" / "workload_strict_v2.npz")
    selected.add(code / "results" / "strict" / "processed" / "workload_strict_v2.json")
    add_files(selected, code / "config", ("*.json",))
    # Generated request arrays exceed 7 GB.  They are deterministic derivatives
    # of the included public-source archive, trace builders, seeds, and
    # manifests.  Retain every manifest and registry without duplicating those
    # rebuildable arrays in the portable artifact.
    traces = code / "results" / "strict" / "traces_v3"
    add_files(selected, traces, ("*.json", "*/manifest.json"))
    for campaign_name in (
        "campaign_formula_v7_20260914",
        "network_stress_formula_v4_20260914",
        "load_scaling_formula_v2_20260914",
        "deadline_guard_network_stress_v4_20260914",
        "breakdown_formula_v1_20260914",
        "exact_benchmark_suite_v2_20260914",
        "measured_startup_strict_v1_20260914",
        "paper_tables_formula_v3_20260914",
        "paper_figures_formula_v5_20260915",
    ):
        add_files(selected, code / "results" / "strict" / campaign_name, ("**/*",))
    for strict_file in (
        "implementation_conformance_final_v16_20260914.json",
        "unit_tests_final_20260915.xml",
        "submission_consistency_audit_final_20260916.json",
    ):
        selected.add(code / "results" / "strict" / strict_file)
    add_files(selected, code / "results" / "revision", (
        "forecast_prequential_20260905/*", "citation_audit_20260913/*",
        "PHYSICAL_FIGURE_CONTRACT.md", "physical_sampling_audit_v3_20260906.json",
        "MEASURED_STARTUP_FIGURE_CONTRACT.md",
        "audits/reviewer_response_audit_20260916.json",
    ))
    figures = code / "results" / "revision" / "figures"
    add_files(selected, figures, (
        "FIGURE_CONTRACT.md", "calibration_*", "physical_calibration.*",
        "fig_physical_scheduler*", "fig_container_startup*",
    ))

    physical = code / "results" / "revision" / "physical"
    add_files(selected, physical / "portable_model_20260905", ("*",))
    add_files(selected, physical / "inference_calibration_v2_20260905",
              ("summary.json", "service_models.json", "timing_records.csv", "provenance.json",
               "port_*_summary.json"))
    add_files(selected, physical / "inference_calibration_port04_repeat_20260909",
              ("summary.json", "service_models.json", "timing_records.csv", "provenance.json",
               "port_04_summary.json", "port_04.jsonl", "REPEAT_PROTOCOL.md"))
    add_files(selected, physical / "activation_transport_isolated_20260905",
              ("activation_transport_analysis.json", "protocol.json", "summary.json",
               "transport_timing_records.csv", "port_*_transport_summary.json"))
    add_files(selected, physical / "power_thermal_20260905",
              ("summary.json", "analysis/**/*", "*.csv"))
    scheduler = physical / "physical_scheduler_full_v3_complete_20260906"
    add_files(selected, scheduler, ("manifest.json", "campaign_status.json",
                                    "archive_verification.json", "analysis/**/*"))
    container = physical / "container_startup_full_v6_complete_20260906"
    add_files(selected, container, ("manifest.json", "campaign_status.json",
                                    "archive_verification.json", "analysis/**/*",
                                    "source_snapshot/*.py"))
    # Raw container timings and acquisition records are compact and contain no
    # endpoint addresses or authentication material.
    add_files(selected, container, ("port_*/activation_*.json",
                                    "port_*/runtime_evidence/*_acquisition.json"))

    for name in ("main.tex", "main_clean.tex", "response.tex",
                 "supplement.tex", "supplement.aux", "assess.tex", "refs.bib", "main.pdf", "main_clean.pdf",
                 "response.pdf", "supplement.pdf", "assess.pdf", "document_qa.json",
                 "build_working_audit.json", "SUBMISSION_POLICY_AUDIT.md"):
        selected.add(manuscript / name)
    section_names = (
        "paper_values_macros.tex", "introduction.tex", "related_work.tex", "model.tex",
        "algorithm.tex", "evaluation.tex", "conclusion.tex", "data_availability.tex",
        "ai_disclosure.tex", "platform_results.tex", "nominal_results.tex",
        "sampling_results.tex", "load_scaling_results.tex", "breakdown_results.tex",
        "cache_capacity_results.tex", "component_ablation_results.tex",
        "energy_pressure_results.tex", "tenant_isolation_results.tex",
        "request_flood_results.tex", "power_uncertainty_results.tex",
        "network_stress_results.tex", "deadline_guard_results.tex",
        "exact_benchmark_results.tex", "calibration_table.tex", "thermal_fit_table.tex",
        "physical_calibration_caption.tex", "power_calibration_table.tex",
        "container_activation_table.tex", "measured_startup_table.tex",
        "physical_poe_scheduler_table.tex", "physical_fleet_scheduler_table.tex",
    )
    selected.update(manuscript / "sections" / name for name in section_names)
    selected.update(manuscript / "responses" / name for name in (
        "associate_editor.tex", "reviewer1.tex", "reviewer2.tex", "reviewer3.tex"
    ))
    current_figure_names = tuple(
        name
        for stem in (
            "fig1_architecture", "fig2_primary", "fig3_sampling_cache",
            "fig4_effects_stress", "figS_load_scaling", "figS_breakdown",
        )
        for name in ((f"{stem}.pdf", f"{stem}.svg", f"{stem}.png")
                     if stem == "fig1_architecture"
                     else (f"{stem}.pdf", f"{stem}.svg", f"{stem}.png", f"{stem}_source.csv"))
    )
    selected.update(manuscript / "figures" / name for name in current_figure_names)
    selected.add(manuscript / "figures" / "figure_provenance.json")
    add_files(selected, manuscript / "figures", (
        "physical_calibration.*", "calibration_*",
        "fig_container_startup*", "fig_physical_scheduler*",
    ))
    selected.add(PACKAGE / "REVISION_STATUS.md")
    missing = [path for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sorted(selected, key=lambda path: str(path).lower())


def relative(path: Path) -> Path:
    # Preserve the trace registries at their executable locations, but archive
    # per-trace manifests outside the live trace tree.  A manifest-only trace
    # directory is not runnable and would prevent the deterministic builders
    # from reconstructing the omitted multi-gigabyte NumPy arrays.
    trace_root = PACKAGE / "code" / "results" / "strict" / "traces_v3"
    if path.name == "manifest.json" and trace_root in path.parents:
        return (
            Path("code/results/strict/trace_manifests_v3")
            / path.relative_to(trace_root)
        )
    return path.relative_to(PACKAGE)


def sanitize_text(content: str) -> tuple[str, list[str]]:
    transformations = []
    content, count = PRIVATE_IPV4.subn("[REDACTED-LAN-IP]", content)
    if count:
        transformations.append(f"private_ipv4:{count}")
    content, count = MAC_ADDRESS.subn("[REDACTED-MAC]", content)
    if count:
        transformations.append(f"mac_address:{count}")
    return content, transformations


def refresh_internal_provenance(destination: Path) -> set[Path]:
    """Rebind nested hashes after deterministic path sanitization.

    The release manifest always hashes the post-sanitization bytes.  This
    function additionally updates the paper-value and submission-audit hashes
    that point to sanitized protocol or value files, so the portable artifact
    remains internally closed rather than merely having a valid outer ZIP.
    """

    changed: set[Path] = set()
    strict = destination / "code" / "results" / "strict"

    # Manifests are archived outside the live trace tree because the rebuildable
    # multi-gigabyte NumPy arrays are intentionally omitted.  Rebind both trace
    # registries to those archived manifests after path sanitization.
    primary_registry_path = strict / "traces_v3" / "primary_trace_registry.json"
    primary_registry = json.loads(primary_registry_path.read_text(encoding="utf-8"))
    for record in primary_registry.get("traces", []):
        directory = str(record["directory"])
        relative_manifest_root = Path("results/strict/trace_manifests_v3") / directory
        manifest_path = destination / "code" / relative_manifest_root / "manifest.json"
        record["path"] = relative_manifest_root.as_posix()
        record["manifest_sha256"] = digest(manifest_path)
    primary_registry_path.write_text(
        json.dumps(primary_registry, indent=2) + "\n", encoding="utf-8"
    )
    changed.add(primary_registry_path)

    load_registry_path = strict / "traces_v3" / "load_scaling_registry.json"
    load_registry = json.loads(load_registry_path.read_text(encoding="utf-8"))
    for record in load_registry.get("traces", []):
        directory = Path(str(record["directory"])).name
        relative_manifest_root = Path("results/strict/trace_manifests_v3") / directory
        manifest_path = destination / "code" / relative_manifest_root / "manifest.json"
        record["directory"] = relative_manifest_root.as_posix()
        record["manifest_sha256"] = digest(manifest_path)
    load_registry_path.write_text(
        json.dumps(load_registry, indent=2) + "\n", encoding="utf-8"
    )
    changed.add(load_registry_path)

    campaign_directories = {
        "main": "campaign_formula_v7_20260914",
        "network": "network_stress_formula_v4_20260914",
        "deadline": "deadline_guard_network_stress_v4_20260914",
        "load": "load_scaling_formula_v2_20260914",
        "breakdown": "breakdown_formula_v1_20260914",
        "measured_startup": "measured_startup_strict_v1_20260914",
    }
    protocol_hashes = {
        name: digest(strict / directory / "protocol.json")
        for name, directory in campaign_directories.items()
    }

    table_root = strict / "paper_tables_formula_v3_20260914"
    paper_values_path = table_root / "paper_values.json"
    paper_values = json.loads(paper_values_path.read_text(encoding="utf-8"))
    for name, directory in campaign_directories.items():
        record = paper_values["protocols"][name]
        record["directory"] = f"results/strict/{directory}"
        record["protocol_sha256"] = protocol_hashes[name]
    paper_values_path.write_text(json.dumps(paper_values, indent=2) + "\n", encoding="utf-8")
    changed.add(paper_values_path)

    table_provenance_path = table_root / "table_provenance.json"
    table_provenance = json.loads(table_provenance_path.read_text(encoding="utf-8"))
    table_provenance["paper_values_sha256"] = digest(paper_values_path)
    for name in table_provenance["outputs"]:
        table_provenance["outputs"][name] = digest(table_root / name)
    table_provenance_path.write_text(
        json.dumps(table_provenance, indent=2) + "\n", encoding="utf-8"
    )
    changed.add(table_provenance_path)

    audit_path = strict / "submission_consistency_audit_final_20260916.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["campaign_protocol_sha256"] = protocol_hashes
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    changed.add(audit_path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise ValueError("Refusing to overwrite an existing release artifact")
    files = selection()
    destination.mkdir(parents=True)
    entries = []
    forbidden = ("password", "private_key", "known_hosts", "network_diagnostics")
    for source in files:
        target_rel = relative(source)
        normalized = target_rel.as_posix().lower()
        if any(token in normalized for token in forbidden):
            raise ValueError("Forbidden credential/network path selected: " + normalized)
        text_type = source.suffix.lower() in {".py", ".tex", ".bib", ".md", ".txt", ".json",
                                                ".csv", ".yaml", ".xml"}
        transformations = []
        if text_type:
            content = source.read_bytes().decode("utf-8", errors="replace")
            private_key_marker = "BEGIN OPENSSH " + "PRIVATE KEY"
            if private_key_marker in content:
                raise ValueError("Private-key material found in selected file: " + normalized)
        target = destination / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if text_type:
            content, transformations = sanitize_text(content)
            if PRIVATE_IPV4.search(content) or MAC_ADDRESS.search(content):
                raise ValueError("LAN identity survived sanitization: " + normalized)
            if transformations:
                with target.open("w", encoding="utf-8", newline="") as handle:
                    handle.write(content)
            else:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        entries.append({"path": target_rel.as_posix(), "bytes": target.stat().st_size,
                        "sha256": digest(target), "transformations": transformations})
    refreshed = refresh_internal_provenance(destination)
    entry_by_path = {row["path"]: row for row in entries}
    for path in refreshed:
        relative_path = path.relative_to(destination).as_posix()
        row = entry_by_path[relative_path]
        row["bytes"] = path.stat().st_size
        row["sha256"] = digest(path)
        row["transformations"] = [*row["transformations"], "internal_provenance_refresh"]
    manifest = {"schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "credential_free_allowlist": True, "file_count": len(entries),
                "total_bytes": sum(item["bytes"] for item in entries), "entries": entries}
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "MANIFEST.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in entries), encoding="utf-8")
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, (Path(destination.name) / path.relative_to(destination)).as_posix())
    verification = {"archive": archive.name, "bytes": archive.stat().st_size,
                    "sha256": digest(archive), "files": len(entries) + 2}
    archive.with_suffix(".zip.sha256.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
