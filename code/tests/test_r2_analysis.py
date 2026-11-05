from copy import deepcopy
import numpy as np
import pytest
from src.analyze_r2_physical import interval,holm
from src.analyze_r2_switch import parse_counters
from src.analyze_r2_physical import failure_intervals
from src.analyze_r2_physical import origin_verifier,sha
from src.r2_evidence_macros import write_macros
from src.physical_trace_audit import audit_trace
from tests.test_physical_analysis import trace_fixture

def test_partial_metrics_do_not_create_selected_complete_case_confidence_intervals():
    mean,half,n=interval([1.]*9+[np.nan])
    assert np.isnan(mean) and np.isnan(half) and n==9
    assert interval([2.]*10)==(2.,0.,10)
    with pytest.raises(ValueError):
        interval([2.]*9)

def test_holm_preserves_all_contrasts_and_step_down_monotonicity():
    rows=[dict(p_unadjusted=p) for p in [.1,.001,.04,1.]]
    holm(rows)
    assert [r['p_holm_primary'] for r in rows]==pytest.approx([.2,.004,.12,1.])

def test_matched_compute_guard_is_audited_from_recorded_parameters():
    events,params,spec=trace_fixture()
    params['energy_guard']=True
    spec['method']='compute_only'
    audit_trace(events,params,spec)
    broken=deepcopy(events)
    next(e for e in broken if e['event']=='state')['credit_j'][0]+=1
    with pytest.raises(ValueError,match='reconstruction'):
        audit_trace(broken,params,spec)

def test_nested_restriction_gap_components_match_total_certificate():
    events,params,spec=trace_fixture()
    audit=audit_trace(events,params,spec)
    assert audit['eligibility_gap_sum_j']>=0
    assert audit['deadline_gap_sum_j']>=0
    assert audit['eligibility_gap_sum_j']+audit['deadline_gap_sum_j']==pytest.approx(audit['restriction_gap_sum_j'])

def test_switch_counter_parser_rejects_missing_totals():
    fixture='100Mbps-speed mode, full-duplex mode\n Input (total):  123 packets, 456 bytes\n Output (total):  321 packets, 654 bytes\n 0 input errors, 1 output errors'
    parsed=parse_counters(fixture)
    assert parsed['line_rate_mbps']==100 and parsed['input_bytes']==456 and parsed['output_packets']==321
    assert parsed['drops'] is None and parsed['output_errors']==1
    with pytest.raises(ValueError,match='total counters'):
        parse_counters(fixture.replace('Output (total)','Output (normal)'))


def test_nested_acquisition_failures_retain_their_actual_time_support():
    failures=[{'collectors':[{'errors':[{'query_start':4.,'query_end':6.}]}]},
              {'latest_midpoint':11.,'monotonic':13.6},
              {'failures':[{'query_start':14.,'query_end':19.}]}]
    assert list(failure_intervals(failures))==[(4.,6.),(11.,13.6),(14.,19.)]


def test_numerical_macros_keep_sign_source_and_undefined_values(tmp_path):
    import json
    tex=tmp_path/'values.tex'
    registry=tmp_path/'registry.json'
    source={'csv':'paired_effects.csv','field':'difference_mean','sign':-1,'scale':100}
    write_macros(tex,registry,[('SuccessDifference',-1.25,2,source),
        ('AdjustedProbability',2.1e-8,None,{'field':'p_holm_primary'}),
        ('MissingEnergy',np.nan,2,{'field':'energy'})])
    text=tex.read_text()
    rows=json.loads(registry.read_text())['macros']
    assert '{-1.25}' in text
    assert r'2.1\times10^{-8}' in text
    assert rows[0]['source']==source and rows[0]['value']==-1.25
    assert rows[2]['value'] is None and 'not available' in rows[2]['display']
    with pytest.raises(ValueError,match='unique'):
        write_macros(tex,registry,[('Duplicate',1,2,{}),('Duplicate',2,2,{})])


def test_public_recovery_metadata_requires_both_registered_origin_and_current_hash(tmp_path):
    import json
    import hashlib
    campaign=tmp_path/'code/results/campaign'
    campaign.mkdir(parents=True)
    record=campaign/'recovery.json'
    old=hashlib.sha256(b'original deployment metadata').hexdigest()
    record.write_text('{"path":"portable/path","count":140}')
    with pytest.raises(ValueError,match='Unverified origin'):
        origin_verifier(campaign)(record,old)
    mapping={'files':[{'path':record.relative_to(tmp_path).as_posix(),
        'original_sha256':old,'public_sha256':sha(record),
        'transformation':'deployment_redaction_or_portable_path_serialization'}]}
    (tmp_path/'PUBLIC_PROVENANCE.json').write_text(json.dumps(mapping))
    verify=origin_verifier(campaign)
    verify(record,old)
    with pytest.raises(ValueError,match='Unverified origin'):
        verify(record,'a'*64)
    record.write_text('{"path":"portable/path","count":139}')
    with pytest.raises(ValueError,match='Unverified origin'):
        verify(record,old)
