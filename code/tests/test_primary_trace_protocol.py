from src.prepare_primary_traces import PROBABILITY, SEEDS, trace_specs


def test_primary_trace_registry_is_complete_and_unique() -> None:
    specs = trace_specs()
    assert len(specs) == 16
    assert len({str(spec["directory"]) for spec in specs}) == 16
    assert PROBABILITY == 4.0e-4
    assert tuple(SEEDS) == tuple(range(20260914, 20260919))

    methods = [str(spec["method"]) for spec in specs]
    releases = [str(spec["release_model"]) for spec in specs]
    assert methods.count("binomial") == 10
    assert methods.count("unbiased_round") == 5
    assert methods.count("deterministic_round") == 1
    assert releases.count("stratified") == 5
    assert releases.count("uniform") == 11
