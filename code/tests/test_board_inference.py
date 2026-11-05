"""Local protocol/safety tests; not substitutes for live-board measurements."""
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import board_inference_worker as worker


def frame(metadata, raw=b''):
    header = json.dumps(metadata).encode()
    return struct.pack('!I', len(header))+header+raw


def decode_frames(data):
    stream = io.BytesIO(data)
    result = []
    while True:
        size = stream.read(4)
        if not size:
            return result
        result.append(json.loads(stream.read(struct.unpack('!I', size)[0])))


class WorkerTests(unittest.TestCase):
    def test_cpu_sensor_excludes_pmic_and_missing_fails_closed(self):
        paths = ['/mock/cpu', '/mock/pmic']
        values = {'/mock/cpu/type': 'CPU-therm', '/mock/cpu/temp': '36500',
                  '/mock/pmic/type': 'PMIC-Die', '/mock/pmic/temp': '100000'}
        with patch.object(worker, 'CPU_PATHS', []), patch.object(worker.glob, 'glob', return_value=paths), \
             patch.object(worker, 'read_text', side_effect=lambda p: values.get(p)):
            self.assertEqual(worker.cpu_temperature(), 36.5)
            self.assertEqual(worker.CPU_PATHS, ['/mock/cpu/temp'])
            for bad in (None, 'nan', 'inf', '200000'):
                values['/mock/cpu/temp'] = bad
                self.assertIsNone(worker.cpu_temperature())

    def test_fragmented_protocol_read_and_eof(self):
        class Fragmented:
            def read(self, n):
                return stream.read(min(n, 3))
        stream = io.BytesIO(b'abcdefgh')
        with patch.object(worker.sys, 'stdin', SimpleNamespace(buffer=Fragmented())):
            self.assertEqual(worker.read_exact(8), b'abcdefgh')
            with self.assertRaises(EOFError):
                worker.read_exact(1)

    def run_worker(self, temperatures, probe=False, repeat=False):
        raw = struct.pack('<132f', *range(132)) + b'padding'
        incoming = (frame(dict(type='init', model={}, temperature_limit_c=65.)) +
                    (frame(dict(type='probe', id='probe')) if probe else b'') +
                    frame(dict(type='infer', id=17, batch=2, payload_bytes=len(raw)), raw) +
                    (frame(dict(type='infer', id=18, batch=2, payload_bytes=len(raw)), raw) if repeat else b'') +
                    frame(dict(type='stop')))
        output = io.BytesIO()
        telemetry = dict(cpu_c=35., sensors={'CPU-therm': 35.}, cpufreq_khz={}, fans_rpm={})
        with patch.object(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(incoming))), \
             patch.object(worker.sys, 'stdout', SimpleNamespace(buffer=output)), \
             patch.object(worker, 'validate_model'), \
             patch.object(worker, 'predict_one', side_effect=lambda m, x: sum(x)), \
             patch.object(worker, 'telemetry', return_value=telemetry), \
             patch.object(worker, 'cpu_temperature', side_effect=temperatures):
            worker.main()
        return decode_frames(output.getvalue())

    def test_full_frame_prediction_order_and_padding(self):
        ready, result, stopped = self.run_worker([35., 35., 35.])
        self.assertEqual(ready['type'], 'ready')
        self.assertEqual(len(ready['source_sha256']), 3)
        self.assertEqual(result['id'], 17)
        self.assertEqual(result['predictions'], [sum(range(66)), sum(range(66, 132))])
        self.assertEqual(result['status'], 'ok')
        self.assertGreaterEqual(result['compute_start'], result['payload_received'])
        self.assertEqual(stopped['type'], 'stopped')

    def test_guard_before_any_compute(self):
        for temperature in (65., 70., None):
            _, result, _ = self.run_worker([temperature])
            self.assertEqual(result['status'], 'temperature_guard')
            self.assertEqual(result['computed'], 0)

    def test_guard_during_batch_preserves_partial_work(self):
        _, result, _ = self.run_worker([35., 65., 65.])
        self.assertEqual(result['status'], 'temperature_guard')
        self.assertEqual(result['computed'], 1)
        self.assertEqual(len(result['predictions']), 1)

    def test_idle_probe_does_not_execute_work(self):
        ready, status, result, stopped = self.run_worker([34., 35., 35., 35.], probe=True)
        self.assertEqual(status['type'], 'status')
        self.assertEqual(status['id'], 'probe')
        self.assertEqual(status['telemetry']['cpu_c'], 34.)
        self.assertEqual(result['computed'], 2)

    def test_guard_latches_for_already_queued_work_even_after_cooling(self):
        _, first, second, _ = self.run_worker([65., 35.], repeat=True)
        self.assertEqual(first['status'], 'temperature_guard')
        self.assertEqual(second['status'], 'temperature_guard')
        self.assertEqual(second['computed'], 0)


if __name__ == '__main__':
    unittest.main()
