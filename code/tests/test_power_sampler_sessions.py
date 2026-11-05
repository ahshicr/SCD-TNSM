import json
import time

import pytest

from src import calibrate_power_thermal as module


class FakeConnection:
    instances = []
    def __init__(self, host):
        self.sock = self
        self.closed = False
        self.count = 0
        self.last_read = {'end_reason': 'matched_prompt'}
        self.instances.append(self)
    def read(self, pattern=None, timeout=8):
        self.count += 1
        if self.count == 1:
            return 'Username:'
        if self.count == 2:
            return 'Password:'
        if self.count == 3:
            return '<H3C>'
        return '\n'.join('GE1/0/%d Enabled Low 3.0 On 4 Delivering' % p
                         for p in range(1,25))+'\n<H3C>'
    def send(self, value):
        if self.closed:
            raise RuntimeError('Use after close')
    def close(self):
        self.closed = True


def wait_samples(sampler, target):
    until = time.monotonic()+2
    while sampler.samples < target and time.monotonic() < until:
        time.sleep(.005)
    assert sampler.samples >= target


def test_planned_renewal_logs_separate_sessions_without_logging_credentials(tmp_path, monkeypatch):
    FakeConnection.instances = []
    monkeypatch.setattr(module, 'Connection', FakeConnection)
    sampler = module.PowerSampler(tmp_path, 'unit-test-only-not-a-real-password', interval=.005,
                                  switch_host='test.invalid', switch_username='test-user')
    try:
        wait_samples(sampler, 2)
        sampler.renew_between_trials()
        assert FakeConnection.instances[0].closed
        wait_samples(sampler, sampler.samples+2)
    finally:
        summary = sampler.close()
    assert summary['thread_stopped'] and not summary['errors'] and summary['sessions'] == 2
    assert sampler._password is None
    rows = [json.loads(line) for line in (tmp_path/'power.jsonl').read_text().splitlines()]
    assert {r['session'] for r in rows} == {0,1}
    assert [r['sample'] for r in rows] == list(range(len(rows)))
    assert all(len(r['ports']) == 22 for r in rows)
    assert all('unit-test-only-not-a-real-password' not in p.read_text() for p in tmp_path.iterdir())
    sessions = [json.loads(line) for line in (tmp_path/'power_sessions.jsonl').read_text().splitlines()]
    assert [s['reason'] for s in sessions] == ['initial_login', 'planned_between_trials']


def test_failed_sampler_is_not_silently_reconnected(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'Connection', FakeConnection)
    sampler = module.PowerSampler(tmp_path, 'unit-test-placeholder', interval=.02,
                                  switch_host='test.invalid', switch_username='test-user')
    try:
        sampler.errors.append({'error': 'test-only failed observation'})
        with pytest.raises(RuntimeError, match='failed/stopped'):
            sampler.renew_between_trials()
    finally:
        sampler.close()
