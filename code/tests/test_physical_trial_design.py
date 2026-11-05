from collections import Counter
from copy import deepcopy
import numpy as np
import pytest
from src.physical_trial_design import design, jobs_for_trial, canonical_hash


def inputs():
    count = np.arange(1, 60*16*4+1, dtype=float).reshape(60, 16, 4)
    aggregate = {'count': count}
    forecast = dict(model_origin=np.zeros(60, dtype=int),
                    prediction=np.ones((60, 16))*2,
                    normalization_scale=np.ones((60, 16)))
    return aggregate, forecast


def test_formal_design_has_280_trials_and_ten_matched_blocks():
    trials = design()
    assert len(trials) == 280
    counts = Counter((t['cohort'], t['scenario'], t['method']) for t in trials)
    assert set(counts.values()) == {10}
    assert all(t['duration_s']==60 and t['drain_s']==15 and t['idle_s']==20 for t in trials)
    assert all(t['hardware_guard_c']==65 and t['max_inflight_per_board']==4 for t in trials)
    assert trials == design()
    with pytest.raises(ValueError):
        design(1)


def test_policies_receive_identical_offered_work_and_no_future_model():
    aggregate, forecast = inputs()
    spec = dict(design()[0], trace_start=0, jobs=720)
    first = jobs_for_trial(spec, aggregate, forecast)
    second = jobs_for_trial(dict(spec, method='scd'), aggregate, forecast)
    assert canonical_hash(first) == canonical_hash(second)
    assert len(first) == 720 and [j['id'] for j in first] == list(range(720))
    assert np.all(np.diff([j['release_s'] for j in first]) >= 0)
    assert all(0 < j['release_s'] < 60 and j['batch']*66*4 <= j['payload_bytes'] <= 4194304 for j in first)
    bad = deepcopy(forecast)
    bad['model_origin'][:] = 100
    with pytest.raises(ValueError, match='future'):
        jobs_for_trial(spec, aggregate, bad)


def test_network_pressure_changes_payload_not_releases_classes_or_deadlines():
    aggregate, forecast = inputs()
    spec = dict(design()[0], trace_start=0, jobs=720, scenario='nominal')
    ordinary = jobs_for_trial(spec, aggregate, forecast)
    heavy = jobs_for_trial(dict(spec, scenario='network_pressure'), aggregate, forecast)
    assert any(a['payload_bytes'] != b['payload_bytes'] for a, b in zip(ordinary, heavy))
    for a, b in zip(ordinary, heavy):
        assert {k:v for k,v in a.items() if k!='payload_bytes'} == {k:v for k,v in b.items() if k!='payload_bytes'}
