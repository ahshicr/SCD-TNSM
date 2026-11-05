"""Repeated real read-only session-renewal validation; no board workload."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import time

from src.calibrate_power_thermal import PowerSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sessions', type=int, default=30)
    parser.add_argument('--samples-per-session', type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    password = getpass.getpass('Switch password (hidden): ')
    sampler, observations = None, []
    result = dict(complete=False, purpose='Management transport validation, not scheduler experiment')
    try:
        sampler = PowerSampler(args.out, password, interval=.05)
        password = None
        for session in range(args.sessions):
            if session:
                sampler.renew_between_trials()
            before = sampler.samples
            deadline = time.monotonic()+90
            while sampler.samples-before < args.samples_per_session:
                if sampler.errors:
                    raise RuntimeError('Power collection failed during validation')
                if time.monotonic() > deadline:
                    raise RuntimeError('Management validation session timed out')
                time.sleep(.05)
            observations.append(dict(session=session, samples=sampler.samples-before))
            print(json.dumps(observations[-1]), flush=True)
        result['complete'] = True
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        password = None
        result.update(utc=datetime.now(timezone.utc).isoformat(), sessions=observations,
                      sampler=sampler.close() if sampler else None)
        if sampler:
            counts = Counter()
            for line in (args.out/'power.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if len(row['ports']) != 22:
                    raise ValueError('Incomplete recorded power table')
                counts[row['session']] += 1
            result['recorded_per_session'] = dict(counts)
            result['complete'] = (result['complete'] and len(counts) == args.sessions and
                min(counts.values()) >= args.samples_per_session and
                result['sampler']['thread_stopped'] and not result['sampler']['errors'])
        (args.out/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
