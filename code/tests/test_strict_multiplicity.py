"""Regressions for retaining the registered family, including null outcomes."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from src.summarize_strict_campaign import paired_row, holm


def row(values):
    n = len(values)
    return paired_row(pd.DataFrame(dict(seed=range(n), metric=np.zeros(n))),
                      pd.DataFrame(dict(seed=range(n), metric=values)),
                      family='test',contrast='paired',reference='ref',challenger='alt',metric='metric')


def test_all_zero_contrast_remains_in_holm_family():
    p = row([0., 0., 0., 0., 0.])['p_unadjusted']
    assert p == 1.
    # Omitting this registered null would turn .03 into a false .05 discovery.
    assert holm(np.array([.03, p])).tolist() == pytest.approx([.06, 1.])


def test_no_service_contrast_is_retained_without_a_fabricated_estimate():
    result = row([np.nan] * 5)
    assert result is not None and result['paired_n'] == 0
    assert np.isnan(result['difference_challenger_minus_reference'])
    assert result['p_unadjusted'] == 1.


def test_near_constant_effect_uses_actual_variation():
    values = np.array([1., 1., 1., 1., 1.000001])
    p = row(values)['p_unadjusted']
    assert p > 0
    assert p == pytest.approx(ttest_1samp(values, 0.).pvalue, rel=1e-12, abs=0)


def test_single_reconstruction_does_not_claim_inference():
    result = row([.3])
    assert result['paired_n'] == 1
    assert np.isnan(result['p_unadjusted'])
    assert np.isnan(result['ci95_lower'])
