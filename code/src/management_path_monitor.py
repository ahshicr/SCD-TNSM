"""Read-only two-endpoint reachability log during a registered recovery.

One ICMP echo per second from the controller and one from port 1 are retained.
The monitor does not alter trial acceptance, scheduler inputs or power values.
It provides independent evidence if the management connection becomes silent.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from src.board_capabilities import connect


class ManagementPathMonitor:
    def __init__(self, directory, row, key, host):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.remote = connect(row, key)
        self.local = None
        self.threads = []
        self.errors = []
        self.stop = threading.Event()
        command = 'ping -D -n -i 1 ' + shlex.quote(host)
        self.remote_in, self.remote_out, self.remote_err = self.remote.exec_command(command)
        self.local = subprocess.Popen(['ping', '-t', '-w', '900', host],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Management reachability diagnostics only, excluded from performance and power estimands.',
            endpoints=['controller', 'port_1'], interval_s=1,
            intervention='Two additional ICMP echoes per second. No switch setting or scheduler input changes.',
            selection='All replacement trials use the same monitor. Failures cannot relax acceptance rules.')
        (self.directory/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        for label, stream, codec in [('controller', self.local.stdout, 'gbk'),
                                     ('port_1', self.remote_out, 'utf-8')]:
            thread = threading.Thread(target=self._collect, args=(label, stream, codec), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _collect(self, label, stream, codec):
        try:
            with (self.directory/(label+'.jsonl')).open('x', encoding='utf-8') as log:
                while not self.stop.is_set():
                    line = stream.readline()
                    if not line:
                        break
                    text = line.decode(codec, 'replace') if isinstance(line, bytes) else line
                    log.write(json.dumps(dict(monotonic=time.monotonic(),
                        utc=datetime.now(timezone.utc).isoformat(), text=text.strip()), ensure_ascii=False)+'\n')
                    log.flush()
        except Exception as exc:
            if not self.stop.is_set():
                self.errors.append(dict(endpoint=label, error=repr(exc)))

    def close(self):
        self.stop.set()
        if self.local.poll() is None:
            self.local.terminate()
        self.local.wait(timeout=5)
        self.remote_out.channel.close()
        self.remote.close()
        for thread in self.threads:
            thread.join(timeout=5)
        self.local.stdout.close()
        result = dict(complete=True, errors=self.errors,
            threads_stopped=all(not thread.is_alive() for thread in self.threads))
        (self.directory/'completion.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
