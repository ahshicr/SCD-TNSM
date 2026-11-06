"""Boundary regressions for no-service runs and complete paired replication."""
import numpy as np
import pandas as pd
import pytest
from src.summarize_strict_campaign import (
    ADMISSION_CONDITIONAL_METRICS, normalize_conditional_metrics, mean_ci, paired_row,
)


def test_zero_admission_is_undefined_only_for_conditional_outcomes():
    raw = pd.DataFrame(dict(admitted_requests=[0, 5], success_ratio=[0., 0.],
                            jain_tenant_success=[0., 0.], rejection_fraction=[1., .5]))
    for name in ADMISSION_CONDITIONAL_METRICS:
        raw[name] = [1. if 'miss_ratio' in name else 0., 0.]
    before = raw.copy(deep=True)
    result = normalize_conditional_metrics(raw)
    assert result.loc[0, list(ADMISSION_CONDITIONAL_METRICS)].isna().all()
    assert (result.loc[1, list(ADMISSION_CONDITIONAL_METRICS)] == 0).all()
    pd.testing.assert_frame_equal(raw,before)
    for name in ['success_ratio','jain_tenant_success','rejection_fraction']:
        pd.testing.assert_series_equal(result[name],raw[name])


def test_no_admission_pair_has_no_numeric_effect_but_stays_registered():
    frame = pd.DataFrame(dict(seed=range(5),admitted_requests=[0]*5,miss_ratio=[1.]*5))
    report = normalize_conditional_metrics(frame)
    row = paired_row(report,report,family='energy',contrast='paired',reference='scd',
                     challenger='cf',metric='miss_ratio')
    assert row['paired_n'] == 0 and row['p_unadjusted'] == 1.
    assert np.isnan(row['difference_challenger_minus_reference'])
    assert np.isnan(row['ci95_lower']) and np.isnan(row['ci95_upper'])


def test_partly_missing_replicates_are_not_silently_selected():
    with pytest.raises(ValueError,match='Incomplete finite replicate'):
        mean_ci([0., .1, np.nan, .2, .3])


@pytest.mark.parametrize('invalid',[-1., .5, np.inf, np.nan])
def test_invalid_denominator_stops_reporting(invalid):
    with pytest.raises(ValueError,match='Invalid admission denominator'):
        normalize_conditional_metrics(pd.DataFrame(dict(admitted_requests=[invalid],miss_ratio=[1.])))
