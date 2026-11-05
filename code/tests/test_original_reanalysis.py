from pathlib import Path
import pytest
from reanalyse_original_hardware import OutputView,numeric_equal,register_hashes


def test_output_redirection_keeps_acquisition_reads_unchanged(tmp_path):
    original=tmp_path/'raw'
    fresh=tmp_path/'reanalysis'
    view=OutputView(original,fresh,{'analysis','service_models.json'})
    assert view/'trial_000'==original/'trial_000'
    assert view/'analysis'==fresh/'analysis'
    assert view/'service_models.json'==fresh/'service_models.json'
    assert Path(view)==original
    with pytest.raises(ValueError,match='outside'):
        OutputView(original,original/'outputs',{'analysis'})


def test_numerical_comparison_accepts_only_verified_provenance_hash_changes():
    register_hashes('a'*64,'b'*64)
    register_hashes('b'*64,'c'*64)
    numeric_equal({'hash':'a'*64,'method':'SCD','results':[1,2.5,None,True]},
                  {'hash':'c'*64,'method':'SCD','results':[1,2.5,None,True]})
    with pytest.raises(AssertionError,match='scientific label'):
        numeric_equal({'method':'SCD'},{'method':'Compute-first'})
    with pytest.raises(AssertionError,match='unverified metadata'):
        numeric_equal({'hash':'a'*64},{'hash':'d'*64})
    with pytest.raises(AssertionError,match='numerical'):
        numeric_equal({'results':[1,2.5]},{'results':[1,2.6]})
    with pytest.raises(AssertionError,match='shape'):
        numeric_equal([1,2],[1])
