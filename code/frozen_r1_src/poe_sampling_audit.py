"""Time-stamped read-only PoE telemetry with explicit measurement boundary.

Only reports ports supplying power. Off ports are missing energy measurements,
not zero-consumption endpoints. The router at port 2 is always excluded.
"""
import argparse
import csv
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import time

from src.switch_transport import Connection


def parse_ports(output):
    result=[]
    for port,watts,state in re.findall(r'GE1/0/(\d+)\s+Enabled\s+\S+\s+([\d.]+)\s+(On|Off)',output):
        if int(port) in (2,15): continue
        result.append({'port':int(port),'poe_on':state=='On',
                       'poe_output_w':float(watts) if state=='On' else None})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=120)
    p.add_argument('--interval',type=float,default=1.0)
    p.add_argument('--switch-host',default=os.environ.get('EDGE_SWITCH_HOST'))
    p.add_argument('--switch-username',default=os.environ.get('EDGE_SWITCH_USERNAME'))
    args=p.parse_args()
    if not args.switch_host or not args.switch_username:
        raise ValueError('Supply EDGE_SWITCH_HOST and EDGE_SWITCH_USERNAME')
    args.out.mkdir(parents=True,exist_ok=False)
    password=getpass.getpass('Switch password (hidden): ')
    prompt=r'(?:^|[\r\n])\s*<[^\r\n>]+>\s*$'
    conn=None
    try:
        conn=Connection(args.switch_host)
        if not re.search(r'(?:login|username)\s*:',conn.read(r'(?:login|username)\s*:'),re.I):
            raise RuntimeError('Missing username prompt')
        conn.send(args.switch_username)
        if not re.search(r'password\s*:',conn.read(r'password\s*:'),re.I):
            raise RuntimeError('Missing password prompt')
        conn.send(password)
        if not re.search(prompt,conn.read(prompt)): raise RuntimeError('Login failed')
        fields=['sample','query_start_utc','query_elapsed_s','port','poe_on','poe_output_w']
        with (args.out/'samples.csv').open('w',newline='',encoding='utf-8') as f, (args.out/'raw.jsonl').open('w',encoding='utf-8') as raw:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            clock=time.monotonic()
            for sample in range(args.samples):
                remaining=clock+sample*args.interval-time.monotonic()
                if remaining>0: time.sleep(remaining)
                timestamp=datetime.now(timezone.utc).isoformat()
                started=time.monotonic()
                conn.send('display poe interface')
                reply=conn.read(prompt,8).replace(password,'[REDACTED]').replace('\x00','')
                elapsed=time.monotonic()-started
                if not re.search(prompt,reply): raise RuntimeError('Incomplete read')
                ports=parse_ports(reply)
                if len(ports)!=22: raise RuntimeError('Incomplete port table')
                raw.write(json.dumps({'sample':sample,'query_start_utc':timestamp,'query_elapsed_s':elapsed,'output':reply})+'\n');raw.flush()
                writer.writerows(dict(sample=sample,query_start_utc=timestamp,query_elapsed_s=elapsed,**row) for row in ports);f.flush()
                if sample%20==0: print(json.dumps({'sample':sample,'poe_board_ports':[r['port'] for r in ports if r['poe_on']],'query_elapsed_s':elapsed}),flush=True)
        conn.send('quit')
    finally:
        if conn: conn.sock.close()
        password=None


if __name__=='__main__':main()
