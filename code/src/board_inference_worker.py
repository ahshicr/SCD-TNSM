"""Persistent SSH-stdio inference worker; user-space only, no open LAN listener.

Binary protocol: 4-byte big-endian JSON length, JSON metadata, then payload_bytes
raw bytes for requests. Responses are length-prefixed JSON with no binary tail.
Only the main compute thread writes stdout. A bounded receiver queue allows
transfer of following inputs while a previous input is computing.
"""
from __future__ import print_function
import glob
import hashlib
import json
import math
import os
import platform
import queue
import struct
import sys
import threading
import time
from portable_inference import predict_one, validate_model


def read_exact(n):
    result = bytearray()
    while len(result) < n:
        part = sys.stdin.buffer.read(n-len(result))
        if not part:
            raise EOFError()
        result.extend(part)
    return bytes(result)


def receive_header(limit=4*1024*1024):
    size, = struct.unpack('!I', read_exact(4))
    if size > limit:
        raise ValueError('Header exceeds protocol limit')
    return json.loads(read_exact(size).decode('utf-8'))


def send(obj):
    data = json.dumps(obj, separators=(',', ':'), allow_nan=False).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('!I', len(data))+data)
    sys.stdout.buffer.flush()


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip().replace('\x00', '')
    except (OSError, IOError):
        return None


def telemetry():
    sensors = {}
    cpu = []
    for path in glob.glob('/sys/class/thermal/thermal_zone*'):
        name, value = read_text(path+'/type'), read_text(path+'/temp')
        if value is None or name is None:
            continue
        try:
            value = float(value)/1000.
        except ValueError:
            continue
        sensors[name] = value
        lower = name.lower()
        if ('cpu' in lower or 'soc' in lower) and 'pmic' not in lower:
            cpu.append(value)
    # A missing identified CPU/SoC sensor disables work rather than treating it
    # as zero temperature or using Nano's unrelated PMIC-Die channel.
    frequencies = {os.path.basename(p): read_text(p+'/scaling_cur_freq')
                   for p in glob.glob('/sys/devices/system/cpu/cpufreq/policy*')}
    fans = {p: read_text(p) for p in glob.glob('/sys/class/hwmon/hwmon*/fan*_input')}
    return dict(cpu_c=max(cpu) if cpu else None, sensors=sensors,
                cpufreq_khz=frequencies, fans_rpm=fans, sampled_monotonic=time.monotonic())


CPU_PATHS = []


def cpu_temperature():
    if not CPU_PATHS:
        for p in glob.glob('/sys/class/thermal/thermal_zone*'):
            name = (read_text(p+'/type') or '').lower()
            if ('cpu' in name or 'soc' in name) and 'pmic' not in name:
                CPU_PATHS.append(p+'/temp')
    values = []
    for p in CPU_PATHS:
        try:
            value = float(read_text(p))/1000.
            if not math.isfinite(value) or not -20. <= value <= 130.:
                return None
            values.append(value)
        except (TypeError, ValueError):
            return None
    return max(values) if values else None


def main():
    init = receive_header()
    if init.get('type') != 'init':
        raise ValueError('Expected init')
    began = time.monotonic()
    model = init['model']
    validate_model(model)
    limit = float(init.get('temperature_limit_c', 65.))
    if not math.isfinite(limit) or not 20. <= limit <= 70.:
        raise ValueError('Invalid temperature safety limit')
    initial_telemetry = telemetry()
    source_hashes = {}
    for name in ('board_inference_worker.py', 'portable_inference.py', 'r2_portable_models.py'):
        with open(os.path.join(os.path.dirname(__file__), name), 'rb') as handle:
            source_hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    send(dict(type='ready', pid=os.getpid(), python=sys.version, arch=platform.machine(),
              board=read_text('/proc/device-tree/model'), telemetry=initial_telemetry,
              source_sha256=source_hashes,
              validation_s=time.monotonic()-began, temperature_limit_c=limit))
    jobs = queue.Queue(maxsize=4)
    latest_detail = [initial_telemetry]
    stop_sampling = threading.Event()

    def sampler():
        while not stop_sampling.wait(1.):
            latest_detail[0] = telemetry()

    sampling_thread = threading.Thread(target=sampler)
    sampling_thread.daemon = True
    sampling_thread.start()

    def current_telemetry():
        result = dict(latest_detail[0])
        result['cpu_c'] = cpu_temperature()
        result['cpu_sampled_monotonic'] = time.monotonic()
        return result

    def receiver():
        try:
            while True:
                header = receive_header(65536)
                if header.get('type') == 'stop':
                    break
                if header.get('type') == 'probe':
                    stamp = time.monotonic()
                    jobs.put((header, b'', stamp, stamp))
                    continue
                if header.get('type') != 'infer':
                    raise ValueError('Unknown request')
                batch, size = int(header['batch']), int(header['payload_bytes'])
                if not 1 <= batch <= 64 or not batch*66*4 <= size <= 16*1024*1024:
                    raise ValueError('Invalid batch or payload size')
                header['batch'], header['payload_bytes'] = batch, size
                receive_start = time.monotonic()
                raw = read_exact(size)
                received = time.monotonic()
                jobs.put((header, raw, receive_start, received))
        except EOFError:
            pass
        except Exception as exc:
            jobs.put((dict(type='error', error=repr(exc)), None, 0., 0.))
        finally:
            jobs.put(None)

    thread = threading.Thread(target=receiver)
    thread.daemon = True
    thread.start()
    thermal_halted = False
    while True:
        item = jobs.get()
        if item is None:
            break
        header, raw, receive_start, received = item
        if raw is None:
            send(header)
            continue
        if header.get('type') == 'probe':
            send(dict(type='status', id=header['id'], telemetry=current_telemetry(),
                      received=received, sampled=time.monotonic(), thermal_halted=thermal_halted))
            continue
        before = current_telemetry()
        response = dict(type='result', id=header['id'], batch=header['batch'],
                        payload_bytes=len(raw), receive_start=receive_start,
                        payload_received=received, before=before)
        if thermal_halted or before['cpu_c'] is None or before['cpu_c'] >= limit:
            thermal_halted = True
            response.update(status='temperature_guard', predictions=[], computed=0)
            send(response)
            continue
        decode_start = time.monotonic()
        values = struct.unpack('<%df' % (header['batch']*66), raw[:header['batch']*66*4])
        compute_start = time.monotonic()
        predictions = []
        status = 'ok'
        safety_check_s = 0.
        for i in range(header['batch']):
            if i:
                checked = time.monotonic()
                current_temp = cpu_temperature()
                safety_check_s += time.monotonic()-checked
                if current_temp is None or current_temp >= limit:
                    status = 'temperature_guard'
                    thermal_halted = True
                    break
            selected_model = model['services'][header.get('service_id', 'mlp')] if model.get('kind') == 'collection' else model
            predictions.append(predict_one(selected_model, values[i*66:(i+1)*66]))
        compute_end = time.monotonic()
        response.update(status=status, predictions=predictions, computed=len(predictions),
                        decode_start=decode_start, compute_start=compute_start,
                        compute_end=compute_end, safety_check_s=safety_check_s, after=current_telemetry())
        send(response)
    stop_sampling.set()
    send(dict(type='stopped', pid=os.getpid()))


if __name__ == '__main__':
    main()
