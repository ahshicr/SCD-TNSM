"""Authenticated read-only H3C inventory, with no credential persistence."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
from src.switch_transport import Connection

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--username', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    password = getpass.getpass('Switch password (hidden): ')
    conn = Connection(a.host)
    prompt = r'(?:^|[\r\n])\s*<[^\r\n>]+>\s*$'
    try:
        if not re.search(r'(?:login|username)\s*:', conn.read(r'(?:login|username)\s*:'), re.I):
            raise RuntimeError('Username prompt missing')
        conn.send(a.username)
        if not re.search(r'password\s*:', conn.read(r'password\s*:'), re.I):
            raise RuntimeError('Password prompt missing')
        conn.send(password)
        password = None
        if not re.search(prompt, conn.read(prompt)):
            raise RuntimeError('Login failed')
        report = dict(utc=datetime.now(timezone.utc).isoformat(), host=a.host, commands={})
        for command in ['display version', 'display interface brief', 'display mac-address',
                        'display poe interface', 'display interface GigabitEthernet1/0/15']:
            conn.send(command)
            reply = conn.read(prompt, 12)
            report['commands'][command] = reply
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'login_ok': True, 'commands': len(report['commands']), 'output': str(a.out)}))
    finally:
        conn.sock.close()
        password = None

if __name__ == '__main__':
    main()
