from __future__ import annotations

from pathlib import Path

from src.prepare_breakdown_traces import PROBABILITIES, SEEDS, probability_label
from src.run_breakdown_extension import METHODS, formal_tasks


def test_breakdown_grid_is_complete_ordered_and_unique() -> None:
    assert PROBABILITIES == tuple(sorted(PROBABILITIES))
    assert PROBABILITIES[0] == 1.0e-7
    assert PROBABILITIES[-1] == 2.0e-4
    tasks = formal_tasks(Path("results/strict/traces_v3"))
    assert len(tasks) == len(PROBABILITIES) * len(SEEDS) * len(METHODS) == 165
    assert len({task["key"] for task in tasks}) == len(tasks)
    assert {task["method"] for task in tasks} == set(METHODS)


def test_breakdown_grid_uses_native_unweighted_traces() -> None:
    tasks = formal_tasks(Path("results/strict/traces_v3"))
    for probability in PROBABILITIES:
        label = probability_label(probability)
        matches = [task for task in tasks if f"breakdown_p{label}_" in Path(task["trace"]).name]
        assert len(matches) == len(SEEDS) * len(METHODS)
    assert all(task["scenario"]["max_requests"] == 0 for task in tasks)
    assert all(task["scenario"]["start_day"] == 10 for task in tasks)
    assert all(task["scenario"]["end_day_exclusive"] == 14 for task in tasks)
