from __future__ import annotations

"""Reproduce the source-locked request-level revision from its raw input."""

import argparse
import subprocess
import sys
from pathlib import Path


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--archive", type=Path, default=Path("data/azurefunctions_dataset2019.tar.xz"))
    parser.add_argument("--workload", type=Path, default=Path("results/strict/processed/workload_strict_v2.npz"))
    parser.add_argument("--trace-root", type=Path, default=Path("results/strict/traces_v3"))
    parser.add_argument("--deployment", type=Path, default=Path("config/strict_deployment_20260914.json"))
    parser.add_argument("--main", type=Path, default=Path("results/strict/campaign_formula_v7_20260914"))
    parser.add_argument("--network", type=Path, default=Path("results/strict/network_stress_formula_v4_20260914"))
    parser.add_argument("--load", type=Path, default=Path("results/strict/load_scaling_formula_v2_20260914"))
    parser.add_argument("--deadline", type=Path, default=Path("results/strict/deadline_guard_network_stress_v4_20260914"))
    parser.add_argument("--breakdown", type=Path, default=Path("results/strict/breakdown_formula_v1_20260914"))
    parser.add_argument("--measured-startup", type=Path, default=Path("results/strict/measured_startup_strict_v1_20260914"))
    parser.add_argument(
        "--container-summary",
        type=Path,
        default=Path("results/revision/physical/container_startup_full_v6_complete_20260906/analysis/summary.json"),
    )
    parser.add_argument(
        "--activation-summary",
        type=Path,
        default=Path("results/revision/physical/activation_transport_isolated_20260905/activation_transport_analysis.json"),
    )
    parser.add_argument("--exact", type=Path, default=Path("results/strict/exact_benchmark_suite_v2_20260914"))
    parser.add_argument("--tables", type=Path, default=Path("results/strict/paper_tables_formula_v3_20260914"))
    parser.add_argument("--figures", type=Path, default=Path("results/strict/paper_figures_formula_v5_20260915"))
    parser.add_argument("--skip-traces", action="store_true")
    parser.add_argument("--skip-campaigns", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--fresh-output-root",
        type=Path,
        help="write all campaigns and derived outputs below a new directory",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    artifact_mode = (Path(__file__).resolve().parent.parent / "MANIFEST.json").is_file()
    if artifact_mode and args.fresh_output_root is None:
        args.fresh_output_root = Path("results/strict/reproduced")
    if args.fresh_output_root is not None:
        root = args.fresh_output_root
        args.main = root / "main"
        args.network = root / "network"
        args.load = root / "load"
        args.deadline = root / "deadline"
        args.breakdown = root / "breakdown"
        args.measured_startup = root / "measured_startup"
        args.exact = root / "exact"
        args.tables = root / "tables"
        args.figures = root / "figures"

    if not args.workload.is_file():
        run("-m", "src.build_strict_workload", "--archive", str(args.archive), "--out", str(args.workload))

    if not args.skip_traces:
        run("-m", "src.prepare_primary_traces", "--workload", str(args.workload),
            "--trace-root", str(args.trace_root), "--workers", str(args.workers))
        run("-m", "src.prepare_load_scaling_traces", "--workload", str(args.workload),
            "--trace-root", str(args.trace_root))
        run("-m", "src.prepare_breakdown_traces", "--workload", str(args.workload),
            "--trace-root", str(args.trace_root), "--workers", str(args.workers))

    campaigns = (
        ("src.run_strict_campaign", args.main),
        ("src.run_network_stress_extension", args.network),
        ("src.run_load_scaling_extension", args.load),
        ("src.run_deadline_guard_extension", args.deadline),
        ("src.run_breakdown_extension", args.breakdown),
    )
    if not args.skip_campaigns:
        for module, output in campaigns:
            run("-m", module, "--trace-root", str(args.trace_root), "--deployment",
                str(args.deployment), "--out-dir", str(output), "--workers", str(args.workers))
        run(
            "-m", "src.run_measured_startup_strict_extension",
            "--trace-root", str(args.trace_root),
            "--deployment", str(args.deployment),
            "--container-summary", str(args.container_summary),
            "--activation-summary", str(args.activation_summary),
            "--out-dir", str(args.measured_startup),
            "--workers", str(args.workers),
        )

    all_campaigns = (*campaigns, ("src.run_measured_startup_strict_extension", args.measured_startup))

    for _, campaign in all_campaigns:
        run("-m", "src.summarize_strict_campaign", "--campaign", str(campaign),
            "--out", str(campaign / "analysis"))

    if not (args.exact / "exact_benchmark_summary.json").is_file():
        run("-m", "src.run_exact_benchmark_suite", "--trace-root", str(args.trace_root),
            "--deployment", str(args.deployment), "--out", str(args.exact))

    common = (
        "--campaign", str(args.main), "--network", str(args.network),
        "--deadline", str(args.deadline), "--load", str(args.load),
        "--breakdown", str(args.breakdown),
    )
    run("-m", "src.strict_paper_tables", *common,
        "--measured-startup", str(args.measured_startup),
        "--exact", str(args.exact), "--deployment",
        str(args.deployment), "--out", str(args.tables))
    run("-m", "src.strict_paper_figures", *common, "--out", str(args.figures))

    conformance = (
        args.fresh_output_root / "implementation_conformance.json"
        if args.fresh_output_root is not None
        else Path("results/strict/implementation_conformance_final_v16_20260914.json")
    )
    command = ["-m", "src.audit_implementation_conformance", "--trace-root", str(args.trace_root),
               "--deployment", str(args.deployment), "--exact-report",
               str(args.exact / "exact_benchmark_summary.json")]
    for _, campaign in all_campaigns:
        command.extend(("--protocol", str(campaign / "protocol.json")))
    command.extend(("--output", str(conformance)))
    run(*command)

    if not args.skip_tests:
        run("-m", "pytest", "-q", "--junitxml",
            "results/strict/unit_tests_final_20260914.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
