"""Repeated real worker activation and bounded bulk-transport/compute overlap.

Local-disk raw logging; no switch credentials, package installs or OS changes.
Worker activation includes interpreter/model transfer, not container creation.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from src.board_inference_client import BoardClient
from src.board_pipeline import BoardPipeline


def utc():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [r for r in json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
            if r.get('ssh_login') and r['port'] not in (2, 15)]
    model_bytes = (args.model/'model.json').read_bytes()
    model = json.loads(model_bytes)
    validation = np.load(args.model/'validation_inputs.npz')
    features, expected = validation['features'], validation['expected']
    key = os.environ['EDGE_SSH_KEY']
    settings = dict(started_utc=utc(), repetitions=args.repetitions, batch=16,
        payload_bytes=[4224, 65536, 1048576, 4194304], depths=[1, 4], workers=args.workers,
        cooldown_seconds=args.cooldown_seconds, temperature_guard_c=65.,
        transport_scope='Declared worker concurrency; includes controller/SSH and shared LAN contention',
        controller_condition=args.controller_condition,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['calibrate_activation_transport.py', 'board_inference_client.py',
                         'board_pipeline.py', 'board_inference_worker.py', 'portable_inference.py']},
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest())
    (args.out/'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    conditions = [(payload, depth) for payload in settings['payload_bytes'] for depth in settings['depths']]
    random.Random(20260907).shuffle(conditions)

    def activation(row):
        succeeded, failures = 0, []
        with (args.out/('port_%02d_activation.jsonl' % row['port'])).open('x', encoding='utf-8') as log:
            for repetition in range(args.repetitions):
                client = None
                record = dict(port=row['port'], repetition=repetition, started_utc=utc())
                started = time.monotonic()
                try:
                    client = BoardClient(row, key, model)
                    record.update(status='ok', provision_and_activation_s=time.monotonic()-started,
                                  worker_ready=client.ready)
                    succeeded += 1
                except Exception as exc:
                    record.update(status='error', error=repr(exc))
                    failures.append(repr(exc))
                finally:
                    if client:
                        record['cleanup'] = client.close()
                log.write(json.dumps(record)+'\n')
                log.flush()
        print(json.dumps(dict(stage='activation', port=row['port'], completed=succeeded,
                              failures=len(failures))), flush=True)
        return dict(port=row['port'], completed=succeeded, failures=failures)

    def transport(row):
        pipe = None
        report = dict(port=row['port'], complete=False, successful_jobs=0, partial_inferences=0)
        try:
            pipe = BoardPipeline(row, key, model)
            report['worker_ready'] = pipe.client.ready
            # Previously incomplete service calibration is not silently borrowed.
            eligible = [b['port'] for b in json.loads(args.calibration.read_text())['boards'] if b['complete']]
            if row['port'] not in eligible:
                report['excluded_reason'] = 'Prior actual-model calibration thermal-guarded'
                return report
            guarded = False
            ordinal = 0
            with (args.out/('port_%02d_transport.jsonl' % row['port'])).open('x', encoding='utf-8') as log:
                for condition, (payload, depth) in enumerate(conditions):
                    pending = deque()
                    submitted = 0
                    while pending or (submitted < args.repetitions and not guarded):
                        while not guarded and submitted < args.repetitions and len(pending) < depth:
                            indices = (np.arange(16)+submitted*7) % len(features)
                            future = pipe.submit(ordinal, features[indices], payload)
                            pending.append((future, indices, submitted))
                            ordinal += 1
                            submitted += 1
                        future, indices, repetition = pending.popleft()
                        result = future.result(timeout=120)
                        computed = result.get('computed', 0)
                        errors = np.abs(np.asarray(result['predictions'])-expected[indices[:computed]]) / np.maximum(
                            np.abs(expected[indices[:computed]]), 1.) if computed else np.zeros(0)
                        parity_ok = bool(not len(errors) or errors.max() < 1e-4)
                        result.update(port=row['port'], condition=condition, repetition=repetition,
                            pipeline_depth=depth, target_payload_bytes=payload, recorded_utc=utc(),
                            parity_ok=parity_ok, max_relative_error=float(errors.max()) if len(errors) else None)
                        log.write(json.dumps(result, allow_nan=False)+'\n')
                        log.flush()
                        if result['status'] == 'ok' and parity_ok:
                            report['successful_jobs'] += 1
                        else:
                            guarded = True
                            report['partial_inferences'] += computed
                            report['stop_reason'] = result['status'] if parity_ok else 'parity_failure'
                    print(json.dumps(dict(stage='transport', port=row['port'], condition=condition,
                        payload=payload, depth=depth, successful_jobs=report['successful_jobs'],
                        stopped=guarded)), flush=True)
                    if guarded:
                        break
                    time.sleep(args.cooldown_seconds)
            report['complete'] = report['successful_jobs'] == len(conditions)*args.repetitions
        except Exception as exc:
            report['error'] = repr(exc)
        finally:
            if pipe:
                report['cleanup'] = pipe.close()
            (args.out/('port_%02d_transport_summary.json' % row['port'])).write_text(
                json.dumps(report, indent=2), encoding='utf-8')
        return report

    with ThreadPoolExecutor(args.workers) as pool:
        activations = [f.result() for f in as_completed([pool.submit(activation, row) for row in rows])]
    with ThreadPoolExecutor(args.workers) as pool:
        transports = [f.result() for f in as_completed([pool.submit(transport, row) for row in rows])]
    summary = dict(started_utc=settings['started_utc'], ended_utc=utc(),
        activation=sorted(activations, key=lambda x: x['port']),
        transport=sorted(transports, key=lambda x: x['port']),
        activation_claim='New remote process and transferred trained model; not container creation, image pull or cold OS cache',
        replication_scope='Twenty within-session observations per activation or payload/depth setting, not scheduling trial CIs')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='activation and transport protocol finished',
        activation_success=sum(x['completed'] for x in activations),
        transport_complete=sum(x['complete'] for x in transports))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    p.add_argument('--calibration', type=Path, default=Path('results/revision/physical/inference_calibration_v2_20260905/summary.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=20)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--cooldown-seconds', type=float, default=20.)
    p.add_argument('--controller-condition', default='Concurrent cohort and six-worker local simulation campaign')
    run(p.parse_args())
