"""Two staggered read-only PSE observation streams with explicit provenance.

Both streams are retained. The optional recovery mode logs every short-timeout
reconnection and stops when observed support is absent for over 2.5 seconds.
Statistical analysis separately verifies every energy interval before integration.
"""
from __future__ import annotations
import heapq
import json
from pathlib import Path
import time
from src.calibrate_power_thermal import PowerSampler


def merge_observations(paths,target):
    def records(path):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                yield (row['query_start']+row['query_end'])/2,row,path.parent.name
    count=0
    previous=float('-inf')
    iterators=[records(path) for path in paths]
    with target.open('x',encoding='utf-8') as stream:
        for midpoint,row,collector in heapq.merge(*iterators,key=lambda r:r[0]):
            if midpoint<=previous:
                raise ValueError('PSE observation clocks are not strictly ordered')
            previous=midpoint
            row=dict(row,collector=collector)
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
            count+=1
    return count


class RedundantPowerSampler:
    def __init__(self,directory,password,recover=False):
        self.directory=Path(directory)
        self._password=password
        self.pair=0
        self.collectors=[]
        self.closed_records=[]
        self.raw_paths=[]
        self.fatal_errors=[]
        self.closed=False
        self.recover=recover
        self._start_pair()

    def _start_pair(self):
        self.collectors=[]
        try:
            for label in ['a','b']:
                if label=='b':
                    time.sleep(.45)
                path=self.directory/('pair_%03d_%s'%(self.pair,label))
                path.mkdir()
                factory=PowerSampler
                if self.recover:
                    from src.recovering_power_sampler import RecoveringPowerSampler
                    factory=RecoveringPowerSampler
                sampler=factory(path,self._password,interval=1.)
                self.collectors.append((path,sampler))
                self.raw_paths.append(path/'power.jsonl')
        except BaseException:
            self._close_pair()
            self._password=None
            raise

    @property
    def errors(self):
        if not self.closed and getattr(self,'recover',False) and self.collectors:
            observed=[sampler.latest_midpoint for _,sampler in self.collectors
                      if sampler.latest_midpoint is not None]
            if observed and time.monotonic()-max(observed)>2.5 and not self.fatal_errors:
                self.fatal_errors.append(dict(error='Observed PSE support gap exceeds 2.5 seconds',
                    pair=self.pair,monotonic=time.monotonic(),latest_midpoint=max(observed)))
        if not self.closed and self.collectors and not any(
                not sampler.errors and sampler.thread.is_alive() for _,sampler in self.collectors):
            if not self.fatal_errors:
                self.fatal_errors.append(dict(error='Both independent PSE observation streams stopped',
                    pair=self.pair,monotonic=time.monotonic(),
                    collectors=[dict(name=path.name,errors=sampler.errors) for path,sampler in self.collectors]))
        return self.fatal_errors

    def _close_pair(self):
        for path,sampler in self.collectors:
            self.closed_records.append(dict(collector=path.name,**sampler.close()))
        self.collectors=[]

    def renew_between_trials(self):
        if self.closed or self.errors:
            raise RuntimeError('Cannot renew a stopped or jointly failed PSE collector')
        self._close_pair()
        self.pair+=1
        self._start_pair()

    def close(self):
        if self.closed:
            raise RuntimeError('PSE collector already closed')
        failure=self.errors[:]
        self._close_pair()
        self.closed=True
        self._password=None
        count=merge_observations(self.raw_paths,self.directory/'power.jsonl')
        result=dict(samples=count,errors=failure,thread_stopped=all(r['thread_stopped'] for r in self.closed_records),
            sessions=len(self.closed_records),collector_pairs=self.pair+1,
            stream_failures=sum(bool(r['errors']) for r in self.closed_records),collectors=self.closed_records,
            boundary='Fixed measured PSE output ports. No external-board or switch AC total is inferred.',
            observation_protocol=('Both staggered 1-Hz streams are retained and merged by query-midpoint time. '
                'Every final energy interval requires observed support within 2.5 s. '+
                ('Every failed query is retained. Unresponsive connections are reauthenticated after a 1-s query timeout. '
                 'An observed support gap over 2.5 s stops the trial.' if self.recover else
                 'No reconnect within a trial. A single failed stream is preserved.')))
        (self.directory/'redundant_acquisition.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
