"""Repeated live CPU-MLP duty plateaus, with read-only PoE and safety guards.

Ten randomized blocks, each containing 60 s at 0/50/100% target duty of one
inference worker. This is calibration, not a policy comparison. The first 15 s
of each plateau are settling observations, retained but excluded from fits.
Actual achieved occupancy is computed from timestamps, not assumed from target.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import time
import numpy as np

from src.board_inference_client import BoardClient
from src.poe_sampling_audit import Connection, parse_ports


def utc():
    return datetime.now(timezone.utc).isoformat()


class PowerSampler:
    def __init__(self, directory, password, interval=1., switch_host=None,
                 switch_username=None):
        self.directory, self.interval = directory, interval
        self.stop = threading.Event()
        self.errors = []
        self.samples = 0
        self.io_lock = threading.RLock()
        self._password = password  # In memory only; cleared on close, never logged.
        self.switch_host = switch_host or os.environ.get('EDGE_SWITCH_HOST')
        self.switch_username = switch_username or os.environ.get('EDGE_SWITCH_USERNAME')
        if not self.switch_host or not self.switch_username:
            raise ValueError('Supply EDGE_SWITCH_HOST and EDGE_SWITCH_USERNAME')
        self.session = 0
        self.prompt = r'(?:^|[\r\n])\s*<[^\r\n>]+>\s*$'
        self.conn = self.login()
        self.session_event('initial_login')
        self.thread = threading.Thread(target=self.sample, name='read-only-poe', daemon=True)
        self.thread.start()

    def login(self):
        conn = Connection(self.switch_host)
        try:
            if not re.search(r'(?:login|username)\s*:', conn.read(r'(?:login|username)\s*:'), re.I):
                raise RuntimeError('Missing username prompt')
            conn.send(self.switch_username)
            if not re.search(r'password\s*:', conn.read(r'password\s*:'), re.I):
                raise RuntimeError('Missing password prompt')
            conn.send(self._password)
            reply = conn.read(self.prompt)
            if not re.search(self.prompt, reply):
                raise RuntimeError('Switch authentication did not reach prompt')
        except BaseException:
            conn.sock.close()
            raise
        return conn

    def session_event(self, reason):
        with (self.directory/'power_sessions.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(session=self.session, reason=reason, utc=utc(),
                monotonic=time.monotonic(), samples_before=self.samples))+'\n')

    def renew_between_trials(self):
        """Planned renewal only outside trials, never recovery through missing data."""
        with self.io_lock:
            if self.errors or self.stop.is_set():
                raise RuntimeError('Cannot renew a failed/stopped sampler')
            replacement = self.login()
            old, self.conn = self.conn, replacement
            old.sock.close()
            self.session += 1
            self.session_event('planned_between_trials')

    def sample(self):
        with (self.directory/'power.jsonl').open('x', encoding='utf-8') as log:
            start = time.monotonic()
            while not self.stop.is_set():
                if self.stop.wait(max(0., start+self.samples*self.interval-time.monotonic())):
                    break
                began = time.monotonic()
                reply = ''
                try:
                    with self.io_lock:
                        began = time.monotonic()
                        self.conn.send('display poe interface')
                        reply = self.conn.read(self.prompt, 8).replace('\x00', '')
                        ended = time.monotonic()
                        ports = parse_ports(reply)
                        if not re.search(self.prompt, reply) or len(ports) != 22:
                            raise RuntimeError('Incomplete PoE table')
                        row = dict(sample=self.samples, session=self.session,
                                   query_start=began, query_end=ended,
                                   recorded_utc=utc(), ports=ports, raw_reply=reply)
                        log.write(json.dumps(row, separators=(',', ':'))+'\n')
                        log.flush()
                        self.samples += 1
                except Exception as exc:
                    failure = dict(time_utc=utc(), error=repr(exc), query_start=began,
                                   query_end=time.monotonic(), raw_reply=reply,
                                   parsed_ports=parse_ports(reply), session=self.session,
                                   transport_diagnostic=getattr(self.conn, 'last_read', None))
                    # A failed read is evidence too. Never interpolate through
                    # it or silently reconnect and label the trial complete.
                    self.errors.append(failure)
                    with (self.directory/'power_failure.json').open('x', encoding='utf-8') as failed:
                        json.dump(failure, failed, indent=2)
                    break

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
        self.conn.sock.close()
        self._password = None
        return dict(samples=self.samples, errors=self.errors,
                    sessions=self.session+1,
                    thread_stopped=not self.thread.is_alive(),
                    boundary='PSE per-port supplied power; excludes switch AC input and external-board supplies')


def run(args):
    if args.blocks < 1 or args.plateau_seconds < 30:
        raise ValueError('Invalid full calibration protocol')
    args.out.mkdir(parents=True, exist_ok=False)
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    eligible = {r['port'] for r in calibration['boards'] if r['complete']}
    rows = json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    rows = [r for r in rows if r['port'] in eligible and r['port'] not in (2, 15)]
    model = json.loads((args.model/'model.json').read_text(encoding='utf-8'))
    examples = np.load(args.model/'validation_inputs.npz')
    features, expected = examples['features'], examples['expected']
    key = os.environ['EDGE_SSH_KEY']
    protocol = dict(started_utc=utc(), temperature_limit_c=65., blocks=args.blocks,
                    plateau_seconds=args.plateau_seconds, settling_seconds=15., batch=16,
                    target_duties=[0., .5, 1.], randomized_order_seed=20260906,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__)]+[Path(__file__).with_name(n) for n in
                       ('board_inference_client.py', 'board_inference_worker.py', 'portable_inference.py', 'poe_sampling_audit.py')]},
                    calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256((args.model/'model.json').read_bytes()).hexdigest(),
                    eligible_ports=sorted(eligible),
                    excluded_ports={str(r['port']): 'Prior complete service calibration unavailable'
                                    for r in calibration['boards'] if not r['complete']},
                    claims='Calibration of achieved single-worker activity, supplied PoE power and measured CPU temperature; no hardware-aging experiment')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    password = getpass.getpass('Switch password (hidden): ')
    sampler = None
    clients = {}
    ready = {}
    stop_load = threading.Event()
    states = {}
    phase = [None]
    futures = []
    pool = ThreadPoolExecutor(max_workers=len(rows))
    try:
        sampler = PowerSampler(args.out, password, switch_host=args.switch_host,
                               switch_username=args.switch_username)
        password = None
        def initialize(row):
            client = BoardClient(row, key, model)
            return row['port'], client
        init = {pool.submit(initialize, row): row['port'] for row in rows}
        for result in as_completed(init):
            port = init[result]
            try:
                port, client = result.result()
                clients[port] = client
                ready[str(port)] = client.ready
            except Exception as exc:
                ready[str(port)] = dict(error=repr(exc))
        (args.out/'worker_ready.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')

        def load(port, client):
            state = dict(port=port, successful_jobs=0, inferred=0, guarded=False, error=None)
            states[port] = state
            last_probe = 0.
            with (args.out/('port_%02d_jobs.jsonl' % port)).open('x', encoding='utf-8') as log, \
                 (args.out/('port_%02d_idle.jsonl' % port)).open('x', encoding='utf-8') as idle:
                while not stop_load.is_set():
                    stage = phase[0]
                    if not stage or stage['duty'] == 0 or time.monotonic() >= stage['end']:
                        if time.monotonic()-last_probe >= 1.:
                            try:
                                record = client.probe('idle-'+str(port)+'-'+str(last_probe))
                                record.update(port=port, recorded_utc=utc())
                                idle.write(json.dumps(record, separators=(',', ':'))+'\n')
                                idle.flush()
                                last_probe = time.monotonic()
                            except Exception as exc:
                                state['error'] = repr(exc)
                                break
                        stop_load.wait(.05)
                        continue
                    ordinal = state['successful_jobs']
                    indices = (np.arange(16)+ordinal*7) % len(features)
                    try:
                        record = client.infer(ordinal, features[indices])
                        record.update(port=port, block=stage['block'], phase=stage['phase'],
                                      duty=stage['duty'], recorded_utc=utc())
                        actual = np.asarray(record['predictions'])
                        reference = expected[indices][:len(actual)]
                        parity = not len(actual) or np.max(np.abs(actual-reference)/np.maximum(np.abs(reference), 1.)) < 1e-4
                        record['parity_ok'] = bool(parity)
                        log.write(json.dumps(record, separators=(',', ':'))+'\n')
                        log.flush()
                        if record['status'] != 'ok' or not parity:
                            state['guarded'] = record['status'] == 'temperature_guard'
                            state['error'] = None if state['guarded'] else record['status']
                            break
                        state['successful_jobs'] += 1
                        state['inferred'] += len(actual)
                        compute = record['compute_end']-record['compute_start']
                        # Target is active worker occupancy, not aggregate four-core CPU utilization.
                        pause = compute*(1./stage['duty']-1.)
                        stop_load.wait(min(pause, max(0., stage['end']-time.monotonic())))
                    except Exception as exc:
                        state['error'] = repr(exc)
                        break
            return state

        futures = [pool.submit(load, port, client) for port, client in clients.items()]
        rng = random.Random(20260906)
        with (args.out/'phases.jsonl').open('x', encoding='utf-8') as log:
            for block in range(args.blocks):
                order = [0., .5, 1.]
                rng.shuffle(order)
                for index, duty in enumerate(order):
                    began = time.monotonic()
                    stage = dict(block=block, phase=index, duty=duty, start=began,
                                 end=began+args.plateau_seconds, recorded_utc=utc())
                    phase[0] = stage
                    log.write(json.dumps(stage)+'\n')
                    log.flush()
                    print(json.dumps(dict(event='plateau', **stage,
                         active_ports=[p for p, s in states.items() if not s['guarded'] and s['error'] is None])), flush=True)
                    while time.monotonic() < stage['end']:
                        if sampler.errors:
                            raise RuntimeError('PoE sampler stopped; preserving incomplete calibration')
                        time.sleep(min(.5, max(0., stage['end']-time.monotonic())))
                    phase[0] = None
        protocol['all_plateaus_finished'] = True
    except Exception as exc:
        protocol['error'] = repr(exc)
        raise
    finally:
        password = None
        phase[0] = None
        stop_load.set()
        for future in futures:
            future.result(timeout=90)
        pool.shutdown(wait=True)
        protocol['power_sampler'] = sampler.close() if sampler else None
        protocol['boards'] = [dict(**states.get(port, dict(port=port)), cleanup=client.close())
                              for port, client in clients.items()]
        protocol['ended_utc'] = utc()
        (args.out/'summary.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--plateau-seconds', type=float, default=60.)
    parser.add_argument('--switch-host', default=os.environ.get('EDGE_SWITCH_HOST'))
    parser.add_argument('--switch-username', default=os.environ.get('EDGE_SWITCH_USERNAME'))
    run(parser.parse_args())
