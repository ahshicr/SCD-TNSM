"""Observable read-only PSE reconnects, never manufactured power samples.

Each attempted query and reconnect failure is retained. A fresh authenticated
connection may replace an unresponsive connection. The enclosing dual-stream
collector still fails if observed support is absent for more than 2.5 seconds.
"""
import json
import re
import time
from src.calibrate_power_thermal import PowerSampler, utc
from src.poe_sampling_audit import parse_ports


class RecoveringPowerSampler(PowerSampler):
    def __init__(self, *args, **kwargs):
        self.recovered_failures = []
        self.latest_midpoint = None
        super().__init__(*args, **kwargs)

    def sample(self):
        consecutive = 0
        attempt = 0
        reconnect = False
        with (self.directory/'power.jsonl').open('x', encoding='utf-8') as log:
            start = time.monotonic()
            while not self.stop.is_set():
                if self.stop.wait(max(0., start+attempt*self.interval-time.monotonic())):
                    break
                attempt += 1
                began = time.monotonic()
                reply = ''
                stage = 'reconnect' if reconnect else 'query'
                try:
                    with self.io_lock:
                        if reconnect:
                            self.conn.sock.close()
                            self.conn = self.login()
                            self.session += 1
                            self.session_event('observed_query_timeout_reconnect')
                            reconnect = False
                        stage = 'query'
                        began = time.monotonic()
                        self.conn.send('display poe interface')
                        reply = self.conn.read(self.prompt, 1.).replace('\x00', '')
                        ended = time.monotonic()
                        ports = parse_ports(reply)
                        if not re.search(self.prompt, reply) or len(ports) != 22:
                            raise RuntimeError('Incomplete PoE table')
                        row = dict(sample=self.samples, attempt=attempt-1, session=self.session,
                            query_start=began, query_end=ended, recorded_utc=utc(),
                            ports=ports, raw_reply=reply)
                        log.write(json.dumps(row, separators=(',', ':'))+'\n')
                        log.flush()
                        self.samples += 1
                        self.latest_midpoint = (began+ended)/2
                        consecutive = 0
                except Exception as exc:
                    failure = dict(time_utc=utc(), error=repr(exc), stage=stage,
                        query_start=began, query_end=time.monotonic(), raw_reply=reply,
                        parsed_ports=parse_ports(reply), session=self.session, attempt=attempt-1,
                        transport_diagnostic=getattr(self.conn, 'last_read', None))
                    self.recovered_failures.append(failure)
                    with (self.directory/'query_failures.jsonl').open('a', encoding='utf-8') as failed:
                        failed.write(json.dumps(failure, separators=(',', ':'))+'\n')
                    consecutive += 1
                    reconnect = True
                    if consecutive >= 3:
                        self.errors.append(dict(error='Three consecutive read-only acquisition failures',
                            failures=self.recovered_failures[-3:]))
                        break

    def close(self):
        result = super().close()
        result['query_failures'] = self.recovered_failures
        result['successful_reconnects'] = self.session
        return result
