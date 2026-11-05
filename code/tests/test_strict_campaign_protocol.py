from __future__ import annotations

import hashlib
from pathlib import Path

from src.audit_implementation_conformance import run_audit
from src.run_deadline_guard_extension import formal_tasks as deadline_tasks
from src.run_measured_startup_strict_extension import formal_tasks as startup_tasks
from src.run_network_stress_extension import formal_tasks as network_tasks
from src.run_strict_campaign import execution_source_hashes, formal_tasks as campaign_tasks
from src.startup_profiles import derive_startup_profiles


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "results" / "strict" / "traces_v3"


def test_formal_campaign_matrices_are_complete_and_unique() -> None:
    startup_evidence = derive_startup_profiles(
        ROOT / "results/revision/physical/container_startup_full_v6_complete_20260906/analysis/summary.json",
        ROOT / "results/revision/physical/activation_transport_isolated_20260905/activation_transport_analysis.json",
    )
    matrices = (
        (campaign_tasks(TRACE_ROOT), 618),
        (network_tasks(TRACE_ROOT), 75),
        (deadline_tasks(TRACE_ROOT), 60),
        (startup_tasks(TRACE_ROOT, startup_evidence), 180),
    )
    for tasks, expected in matrices:
        assert len(tasks) == expected
        assert len({task["key"] for task in tasks}) == expected


def test_execution_hashes_cover_shared_and_runner_sources() -> None:
    runner = ROOT / "src" / "run_network_stress_extension.py"
    hashes = execution_source_hashes(runner)
    assert set(hashes) == {
        "run_network_stress_extension.py",
        "run_strict_campaign.py",
        "scheduler_core.py",
        "strict_config.py",
        "strict_simulate.py",
        "strict_trace.py",
    }
    assert hashes[runner.name] == hashlib.sha256(runner.read_bytes()).hexdigest()


def test_implementation_conformance_without_completed_protocols() -> None:
    reference_arrays = TRACE_ROOT / "binomial_p0004_seed20260914" / "release_s.npy"
    if not reference_arrays.is_file():
        archived = ROOT / "results" / "strict" / "trace_manifests_v3"
        registry = ROOT / "results" / "strict" / "traces_v3" / "primary_trace_registry.json"
        final_report = ROOT / "results" / "strict" / "implementation_conformance_final_v16_20260914.json"
        assert registry.is_file() and final_report.is_file()
        assert len(list(archived.glob("*/manifest.json"))) >= 16
        report = __import__("json").loads(final_report.read_text(encoding="utf-8"))
        assert report["status"] == "PASS" and report["passed"] == report["checks"]
        return
    report = run_audit(
        TRACE_ROOT,
        ROOT / "config" / "strict_deployment_20260914.json",
        [],
    )
    assert report["status"] == "PASS", report["failed_checks"]
    assert report["checks"] >= 13
