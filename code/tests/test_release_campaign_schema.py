import json
import io
import pandas as pd
import pytest
from verify_release import compare_flattened_campaign


def test_nested_scenario_parameters_and_list_outputs_are_fully_checked(tmp_path):
    runs=tmp_path/'runs'
    runs.mkdir()
    record=dict(key='case',success_ratio=.8,tenant_success=[.7,.9],
        scenario_parameters=dict(cache_identity_mode='function',energy_guard=True,cache_ttl_s=None))
    (runs/'case.json').write_text(json.dumps(record))
    reference=pd.read_csv(io.StringIO(pd.json_normalize([record]).to_csv(index=False)))
    compare_flattened_campaign(tmp_path,reference)
    reference['scenario_parameters.cache_identity_mode']='stratum'
    with pytest.raises(AssertionError):
        compare_flattened_campaign(tmp_path,reference)


def test_flattened_campaign_does_not_ignore_missing_reported_columns(tmp_path):
    (tmp_path/'runs').mkdir()
    record=dict(key='case',success_ratio=.8,scenario_parameters=dict(seed=1))
    (tmp_path/'runs/case.json').write_text(json.dumps(record))
    reference=pd.json_normalize([record]).drop(columns='scenario_parameters.seed')
    with pytest.raises(AssertionError,match='schema'):
        compare_flattened_campaign(tmp_path,reference)
