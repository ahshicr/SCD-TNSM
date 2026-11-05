"""Minimal read-only-console Telnet transport, including split pagination prompts.

Adapted from the project's original h3c_telnet_inspect.Connection. TCP reads
are not terminal-record boundaries: both IAC commands and '-- More --' may
be split at ANY byte. Negotiation follows RFC 854; no switch configuration
command is issued by this module. Authentication output is never logged here.
"""
import re
import socket
import time


class Connection:
    def __init__(self, host):
        self.sock = socket.create_connection((host, 23), timeout=5)
        self.sock.settimeout(.4)
        self.pending = b''
        self.enabled_remote = set()

    def read(self, pattern=None, timeout=8):
        text, last_page_start = '', -1
        self.last_read = dict(received_bytes=0, receive_calls=0, end_reason='deadline')
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            try:
                chunk = self.sock.recv(16384)
                self.last_read['receive_calls'] += 1
                self.last_read['received_bytes'] += len(chunk)
                if not chunk:
                    self.last_read['end_reason'] = 'peer_closed'
                    break
            except TimeoutError:
                continue
            data, self.pending = self.pending+chunk, b''
            output, i = bytearray(), 0
            while i < len(data):
                if data[i] != 255:
                    output.append(data[i])
                    i += 1
                    continue
                if i+1 >= len(data):
                    break
                verb = data[i+1]
                if verb == 255:
                    output.append(255)
                    i += 2
                elif verb in (251, 252, 253, 254):
                    if i+2 >= len(data):
                        break
                    option = data[i+2]
                    if verb == 251:
                        if option not in self.enabled_remote:
                            accept = option in (1, 3)
                            self.sock.sendall(bytes([255, 253 if accept else 254, option]))
                            if accept:
                                self.enabled_remote.add(option)
                    elif verb == 252:
                        if option in self.enabled_remote:
                            self.enabled_remote.remove(option)
                            self.sock.sendall(bytes([255, 254, option]))
                    elif verb == 253:
                        self.sock.sendall(bytes([255, 252, option]))
                    i += 3
                elif verb == 250:
                    end = data.find(b'\xff\xf0', i+2)
                    if end < 0:
                        break
                    i = end+2
                else:
                    i += 2
            self.pending = data[i:]
            text += output.decode('utf-8', 'replace')
            # Match accumulated output, not a single recv() fragment. Compare
            # match starts so a growing trailing run of '-' is acknowledged once.
            for match in re.finditer(r'--+\s*more\s*--+', text, re.I):
                if match.start() > last_page_start:
                    self.sock.sendall(b' ')
                    last_page_start = match.start()
            if pattern and re.search(pattern, text, re.I):
                self.last_read['end_reason'] = 'matched_prompt'
                break
        self.last_read['pending_telnet_bytes'] = len(self.pending)
        return text

    def send(self, value):
        self.sock.sendall(value.encode()+b'\r\n')
