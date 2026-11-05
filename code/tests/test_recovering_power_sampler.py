import json
from pathlib import Path
import threading
from types import SimpleNamespace
from src.recovering_power_sampler import RecoveringPowerSampler
from src.redundant_power_sampler import RedundantPowerSampler


def test_failed_query_is_retained_and_not_used_as_an_observation(tmp_path, monkeypatch):
    sampler=RecoveringPowerSampler.__new__(RecoveringPowerSampler)
    sampler.directory=tmp_path
    sampler.interval=0.
    sampler.stop=threading.Event()
    sampler.io_lock=threading.RLock()
    sampler.prompt=r'<H3C>$'
    sampler.samples=0
    sampler.session=0
    sampler.errors=[]
    sampler.recovered_failures=[]
    sampler.latest_midpoint=None
    closed=[]
    attempts=[]
    old=SimpleNamespace(sock=SimpleNamespace(close=lambda:closed.append('old')),
        send=lambda value:None, read=lambda pattern,timeout:'', last_read={'received_bytes':0})
    def read(pattern,timeout):
        assert timeout==1.
        attempts.append('valid')
        if len(attempts)==2:
            sampler.stop.set()
        return '<H3C>'
    fresh=SimpleNamespace(sock=SimpleNamespace(close=lambda:None),send=lambda value:None,read=read)
    sampler.conn=old
    sampler.login=lambda:fresh
    events=[]
    sampler.session_event=events.append
    monkeypatch.setattr('src.recovering_power_sampler.parse_ports',lambda reply:[dict(port=i) for i in range(22)] if reply else [])
    sampler.sample()
    samples=[json.loads(line) for line in (tmp_path/'power.jsonl').read_text().splitlines()]
    failures=[json.loads(line) for line in (tmp_path/'query_failures.jsonl').read_text().splitlines()]
    assert len(samples)==2 and len(failures)==1
    assert all(row['raw_reply']=='<H3C>' and row['session']==1 for row in samples)
    assert failures[0]['raw_reply']=='' and failures[0]['session']==0
    assert sampler.errors==[] and sampler.samples==2 and closed==['old']
    assert events==['observed_query_timeout_reconnect']


def test_recovery_cannot_hide_an_unsupported_measurement_gap(monkeypatch):
    sampler=RedundantPowerSampler.__new__(RedundantPowerSampler)
    sampler.closed=False
    sampler.recover=True
    sampler.pair=2
    sampler.fatal_errors=[]
    stream=SimpleNamespace(latest_midpoint=10., errors=[],thread=SimpleNamespace(is_alive=lambda:True))
    sampler.collectors=[(Path('a'),stream),(Path('b'),stream)]
    monkeypatch.setattr('src.redundant_power_sampler.time.monotonic',lambda:12.6)
    assert sampler.errors[0]['error']=='Observed PSE support gap exceeds 2.5 seconds'
    assert len(sampler.errors)==1
