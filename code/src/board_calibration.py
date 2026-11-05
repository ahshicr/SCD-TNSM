"""Low-duty-cycle CPU-kernel calibration; not a scheduling experiment.

Twenty independent measurements per work size on every pinned board. The
controller records raw observations, rather than substituting old relative
service rates. Stops the probe if CPU temperature reaches 65 C. No disk writes
or package installs occur on boards. Real AI inference is a separate workload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex

from board_capabilities import connect

REMOTE = r'''
import glob,hashlib,json,random,time,sys
def temperature():
    values=[]
    for p in glob.glob('/sys/class/thermal/thermal_zone*'):
        try:
            with open(p+'/type') as f: name=f.read().strip()
            if 'cpu' not in name.lower(): continue
            with open(p+'/temp') as f: values.append(float(f.read())/1000.)
        except (OSError,ValueError): pass
    return max(values) if values else None
sizes=tuple(int(v) for v in sys.argv[1].split(','))
order=[(rep,n) for rep in range(20) for n in sizes]
random.Random(20260905).shuffle(order)
rows=[]
for rep,n in order:
    temp=temperature()
    if temp is None or temp>=65.0:
        print(json.dumps({'rows':rows,'stopped':'temperature_guard','temperature_c':temp}));break
    x=b'tnsm-fixed-calibration-v1'
    t=time.perf_counter()
    for _ in range(n): x=hashlib.sha256(x).digest()
    dt=time.perf_counter()-t
    rows.append({'repetition':rep,'rounds':n,'kernel_s':dt,'temperature_c':temperature(),'digest':x.hex()})
    time.sleep(max(0.20,dt*2.0))
else: print(json.dumps({'rows':rows,'stopped':None}))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ports',type=int,nargs='*')
    p.add_argument('--rounds',default='15000,30000,60000,120000')
    args=p.parse_args()
    nodes=json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    nodes=[r for r in nodes if r['port'] not in (2,15) and r.get('ssh_login') is True]
    if args.ports: nodes=[r for r in nodes if r['port'] in args.ports]
    def run(row):
        result={'port':row['port'],'ip':row['ip']}
        c=None
        try:
            c=connect(row,os.environ['EDGE_SSH_KEY'])
            _,out,err=c.exec_command('python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.rounds),timeout=240)
            value=out.read().decode()
            if out.channel.recv_exit_status(): raise RuntimeError(err.read().decode())
            result.update(json.loads(value))
        except Exception as e: result['error']=repr(e)
        finally:
            if c: c.close()
        print(json.dumps({'port':row['port'],'observations':len(result.get('rows',[])),
                          'stopped':result.get('stopped'),'error':result.get('error')}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=16) as pool: rows=list(pool.map(run,nodes))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'low-duty-cycle infrastructure calibration; not formal replay',
        'kernel':'Python hashlib SHA256 chain; no inference claim','nodes':rows},indent=2),encoding='utf-8')


if __name__=='__main__': main()
