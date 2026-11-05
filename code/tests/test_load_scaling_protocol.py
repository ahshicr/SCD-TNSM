from __future__ import annotations

from pathlib import Path

from src.prepare_load_scaling_traces import LOADS, SEEDS
from src.run_load_scaling_extension import METHODS, formal_tasks


def test_load_protocol_is_complete_and_unique() -> None:
    tasks = formal_tasks(Path("results/strict/traces_v3"))
    assert len(tasks) == len(LOADS) * len(SEEDS) * len(METHODS) == 30
    assert len({task["key"] for task in tasks}) == len(tasks)
    assert {task["method"] for task in tasks} == set(METHODS)
    assert {task["scenario"]["name"] for task in tasks} == {
        "load_0p5",
        "load_2",
    }


def test_load_scaling_changes_native_thinning_probability() -> None:
    probabilities = {record["label"]: record["probability"] for record in LOADS}
    assert probabilities == {"load_0p5": 2.0e-4, "load_2": 8.0e-4}
    tasks = formal_tasks(Path("results/strict/traces_v3"))
    for task in tasks:
        path = Path(task["trace"])
        if task["scenario"]["name"] == "load_0p5":
            assert "binomial_p0002" in path.name
        else:
            assert "binomial_p0008" in path.name
        assert task["scenario"]["max_requests"] == 0
        assert task["scenario"]["start_day"] == 10
        assert task["scenario"]["end_day_exclusive"] == 14
