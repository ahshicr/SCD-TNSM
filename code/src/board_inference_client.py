"""Pinned SSH transport for the persistent standard-library MLP worker."""
from __future__ import annotations
import json
from pathlib import Path
import shlex
import struct
import time

from src.board_capabilities import connect


class BoardClient:
    def __init__(self, row, key, model, temperature_limit_c=65.):
        self.row = row
        self.ssh = connect(row, key)
        self.remote_dir = None
        self.channel = None
        self.remote_files = []
        self.cleanup = dict(stop_acknowledged=False, removed_files=[], directory_removed=False,
                            errors=[])
        try:
            _, stdout, stderr = self.ssh.exec_command('mktemp -d /tmp/tnsm-inference.XXXXXXXX')
            directory = stdout.read().decode().strip()
            if stdout.channel.recv_exit_status() or not directory.startswith('/tmp/tnsm-inference.') or '\n' in directory:
                raise RuntimeError('Cannot create dedicated worker directory')
            self.remote_dir = directory
            with self.ssh.open_sftp() as ftp:
                for name in ('board_inference_worker.py', 'portable_inference.py', 'r2_portable_models.py'):
                    local = Path(__file__).with_name(name)
                    remote = directory+'/'+name
                    ftp.put(str(local), remote)
                    self.remote_files.append(remote)
            self.channel = self.ssh.get_transport().open_session(timeout=10)
            self.channel.settimeout(60)
            begun = time.monotonic()
            command = 'PYTHONDONTWRITEBYTECODE=1 python3 -u '+shlex.quote(directory+'/board_inference_worker.py')
            self.channel.exec_command(command)
            self.send(dict(type='init', model=model, temperature_limit_c=temperature_limit_c))
            self.ready = self.receive()
            self.ready['controller_startup_s'] = time.monotonic()-begun
            if self.ready.get('type') != 'ready':
                raise RuntimeError('Worker did not initialize')
        except BaseException:
            self.close()
            raise

    def send(self, metadata, raw=b''):
        header = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
        # Bulk writes, not one remote interpreter or 32-byte write per job.
        self.channel.sendall(struct.pack('!I', len(header))+header+raw)

    def receive(self):
        def exact(n):
            data = bytearray()
            while len(data) < n:
                part = self.channel.recv(n-len(data))
                if not part:
                    error = self.channel.recv_stderr(8192).decode(errors='replace') if self.channel.recv_stderr_ready() else ''
                    raise EOFError('Worker closed channel: '+error)
                data.extend(part)
            return bytes(data)
        size, = struct.unpack('!I', exact(4))
        if size > 4*1024*1024:
            raise ValueError('Oversized worker response')
        return json.loads(exact(size))

    def probe(self, identity):
        started = time.monotonic()
        self.send(dict(type='probe', id=identity))
        result = self.receive()
        if result.get('type') != 'status' or result.get('id') != identity:
            raise ValueError('Probe response mismatch')
        result.update(controller_send_start=started, controller_received=time.monotonic())
        return result

    def infer(self, job_id, features, payload_bytes=0, service_id='mlp'):
        batch = len(features)
        raw = struct.pack('<%df' % (66*batch), *(float(v) for row in features for v in row))
        raw += bytes(max(0, payload_bytes-len(raw)))
        if len(raw) > 16*1024*1024:
            raise ValueError('Payload exceeds safe worker limit')
        started = time.monotonic()
        self.send(dict(type='infer', id=job_id, batch=batch, payload_bytes=len(raw), service_id=service_id), raw)
        sent = time.monotonic()
        result = self.receive()
        received = time.monotonic()
        if result.get('id') != job_id:
            raise ValueError('Response identity/order mismatch')
        result.update(controller_send_start=started, controller_send_end=sent,
                      controller_received=received, controller_latency_s=received-started)
        return result

    def close(self):
        if self.channel is not None:
            try:
                self.send(dict(type='stop'))
                self.channel.settimeout(15)
                self.cleanup['stop_acknowledged'] = self.receive().get('type') == 'stopped'
            except Exception as exc:
                self.cleanup['errors'].append('stop: '+repr(exc))
            self.channel.close()
            self.channel = None
        if self.remote_dir:
            try:
                with self.ssh.open_sftp() as ftp:
                    for path in self.remote_files:
                        ftp.remove(path)
                        self.cleanup['removed_files'].append(path)
                    ftp.rmdir(self.remote_dir)
                    self.cleanup['directory_removed'] = True
            except Exception as exc:
                # Never recursively delete a directory or mask the main result.
                self.cleanup['errors'].append('cleanup: '+repr(exc))
        self.ssh.close()
        return self.cleanup
