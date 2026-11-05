from __future__ import annotations

from pathlib import Path

from src.strict_config import load_deployment
from src.strict_optimal_benchmark import (
    BenchmarkJob,
    exact_assignment,
    greedy_assignment,
    select_deployment,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_enumeration_dominates_online_greedy() -> None:
    full = load_deployment(ROOT / "config" / "strict_deployment_20260914.json")
    deployment = select_deployment(full, (1, 9))
    jobs = [
        BenchmarkJob(0.0, 0, 0.20, 0.30, 100.0, 100_000.0),
        BenchmarkJob(0.01, 1, 0.05, 0.20, 100.0, 100_000.0),
        BenchmarkJob(0.02, 0, 0.20, 0.30, 100.0, 100_000.0),
        BenchmarkJob(0.03, 1, 0.05, 0.20, 100.0, 100_000.0),
    ]
    exact, enumerated = exact_assignment(jobs, deployment)
    greedy = greedy_assignment(jobs, deployment)
    assert enumerated == 3 ** len(jobs)
    assert exact.feasible and greedy.feasible
    assert exact.objective <= greedy.objective + 1.0e-12
