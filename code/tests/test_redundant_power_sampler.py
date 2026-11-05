import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.redundant_power_sampler import merge_observations,RedundantPowerSampler


def test_redundant_measurements_preserve_every_observation(tmp_path):
    sources=[]
    for label,times in [('a',[1.,2.,3.]),('b',[1.45,2.45,3.45])]:
        path=tmp_path/label/'power.jsonl'
        path.parent.mkdir()
        path.write_text(''.join(json.dumps(dict(query_start=t,query_end=t+.1,ports=[dict(port=1,poe_output_w=t)]))+'\n' for t in times))
        sources.append(path)
    count=merge_observations(sources,tmp_path/'merged.jsonl')
    rows=[json.loads(x) for x in (tmp_path/'merged.jsonl').read_text().splitlines()]
    assert count==6 and [r['query_start'] for r in rows]==[1.,1.45,2.,2.45,3.,3.45]
    assert [r['ports'][0]['poe_output_w'] for r in rows]==[1.,1.45,2.,2.45,3.,3.45]
    assert {r['collector'] for r in rows}=={'a','b'}


def test_joint_failure_is_fatal_but_one_retained_stream_can_continue():
    sampler=RedundantPowerSampler.__new__(RedundantPowerSampler)
    sampler.closed=False
    sampler.fatal_errors=[]
    sampler.pair=3
    good=SimpleNamespace(errors=[],thread=SimpleNamespace(is_alive=lambda:True))
    failed=SimpleNamespace(errors=[dict(error='timeout')],thread=SimpleNamespace(is_alive=lambda:False))
    sampler.collectors=[(Path('a'),good),(Path('b'),failed)]
    assert sampler.errors==[]
    sampler.collectors=[(Path('a'),failed),(Path('b'),failed)]
    assert len(sampler.errors)==1
    assert sampler.errors[0]['pair']==3
    assert len(sampler.errors)==1


def test_redundant_merge_rejects_duplicate_clocks(tmp_path):
    paths=[]
    for name in ['a','b']:
        path=tmp_path/(name+'.jsonl')
        path.write_text(json.dumps(dict(query_start=1.,query_end=1.1))+'\n')
        paths.append(path)
    with pytest.raises(ValueError,match='ordered'):
        merge_observations(paths,tmp_path/'out.jsonl')
