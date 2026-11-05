"""Ten paired blocks on a real 100-Mb/s H3C egress, with disjoint-path controls."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import time
from src.board_capabilities import connect
from src.switch_transport import Connection

class Remote:
    def __init__(self,row,key):
        self.row=row
        self.ssh=connect(row,key)
        _,out,err=self.ssh.exec_command('mktemp -d /tmp/tnsm-udp.XXXXXXXX')
        self.directory=out.read().decode().strip()
        if out.channel.recv_exit_status() or not self.directory.startswith('/tmp/tnsm-udp.') or '\n' in self.directory:
            raise RuntimeError('Dedicated experiment directory unavailable')
        self.path=self.directory+'/worker.py'
        with self.ssh.open_sftp() as ftp:
            ftp.put(str(Path(__file__).with_name('r2_udp_remote.py')),self.path)
    def start(self,spec):
        stdin,stdout,stderr=self.ssh.exec_command('python3 -u '+shlex.quote(self.path),timeout=55)
        stdin.write(json.dumps(spec)+'\n')
        stdin.flush()
        ready=json.loads(stdout.readline())
        if not ready.get('ready'):
            raise RuntimeError('Endpoint did not become ready')
        return stdout,stderr,ready
    def finish(self,operation):
        out,err,ready=operation
        line=out.readline()
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode('utf-8','replace'))
        return dict(ready=ready,result=json.loads(line))
    def close(self):
        with self.ssh.open_sftp() as ftp:
            ftp.remove(self.path)
            ftp.rmdir(self.directory)
        self.ssh.close()

def login(host,username,password):
    connection=Connection(host)
    if not re.search(r'(?:login|username)\s*:',connection.read(r'(?:login|username)\s*:'),re.I):
        raise RuntimeError('No switch username prompt')
    connection.send(username)
    if not re.search(r'password\s*:',connection.read(r'password\s*:'),re.I):
        raise RuntimeError('No switch password prompt')
    connection.send(password)
    prompt=r'(?:^|[\r\n])\s*<[^\r\n>]+>\s*$'
    if not re.search(prompt,connection.read(prompt)):
        raise RuntimeError('Switch login failed')
    return connection,prompt

def snapshot(connection,prompt,commands):
    result={}
    for command in commands:
        start=time.monotonic()
        connection.send(command)
        reply=connection.read(prompt,10).replace('\x00','')
        if not re.search(prompt,reply):
            raise RuntimeError('Incomplete read-only counter response')
        result[command]=dict(query_start=start,query_end=time.monotonic(),reply=reply)
    return result

def design():
    result=[]
    for block in range(10):
        cells=[('shared',0.),('shared',50.),('shared',85.),('shared',105.),('disjoint',85.),('disjoint',105.)]
        random.Random(20261106+block).shuffle(cells)
        for path,load in cells:
            result.append(dict(ordinal=len(result),block=block,path=path,wire_mbps=load,probe_duration_s=30.,
                probe_rate_hz=100.,probe_payload_bytes=256,load_payload_bytes=1400,load_duration_s=34.))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--physical-status',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    frozen={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['run_r2_switch_experiment.py','r2_udp_remote.py','board_capabilities.py','switch_transport.py']}
    trials=design()
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),trials=trials,planned_trials=len(trials),source_sha256=frozen,
        probe_sender_port=1,load_sender_port=3,shared_egress_port=7,disjoint_egress_port=23,
        primary_contrasts=['Shared minus disjoint at 85 Mb/s','Shared minus disjoint at 105 Mb/s'],
        primary_metrics=['P95 completed-probe RTT','probe loss fraction'],
        primary_statistics='Ten paired randomized blocks, Student-t 95 percent CIs, Holm over four prespecified contrasts',
        model_boundary='Direct Ethernet-path contention evidence with switch interface counters. RTT includes both endpoints and is not claimed as internal switch-buffer residence.',
        offered_wire_rate='1400-byte UDP payload plus 66 bytes Ethernet/IP/UDP/FCS/preamble/interframe overhead per packet',
        resource_scope='Only the identified board-to-board paths, no switch configuration changes, no unrelated hosts')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    snap=a.out/'source_snapshot'
    snap.mkdir()
    for name in frozen:
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    password=getpass.getpass('Switch password (hidden): ')
    while not a.physical_status.exists():
        time.sleep(5)
    if not json.loads(a.physical_status.read_text(encoding='utf-8'))['complete']:
        raise RuntimeError('Scheduler campaign incomplete. No overlapping switch experiment started.')
    inventory={r['port']:r for r in json.loads(a.inventory.read_text(encoding='utf-8'))['ports']}
    key=os.environ['EDGE_SSH_KEY']
    commands=['display interface GigabitEthernet1/0/'+str(port) for port in [1,3,7,23]]
    completed=[]
    for spec in trials:
        current={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in frozen}
        if current!=frozen:
            raise RuntimeError('Switch experiment source changed')
        connection,prompt=login(os.environ['EDGE_SWITCH_HOST'],os.environ['EDGE_SWITCH_USERNAME'],password)
        remote={}
        report=dict(spec=spec,complete=False)
        try:
            report['before']=snapshot(connection,prompt,commands)
            for port in [1,3,7,23]:
                remote[port]=Remote(inventory[port],key)
            servers={}
            for port in [7,23]:
                servers[port]=remote[port].start(dict(mode='server',bind_ip=inventory[port]['ip'],bind_port=49665,
                    duration_s=38.,allowed_sources=[inventory[1]['ip'],inventory[3]['ip']]))
            target=7 if spec['path']=='shared' else 23
            load=remote[3].start(dict(mode='load',bind_ip=inventory[3]['ip'],duration_s=34.,
                 target_ip=inventory[target]['ip'],target_port=49665,payload_bytes=1400,wire_mbps=spec['wire_mbps']))
            time.sleep(1.)
            probe=remote[1].start(dict(mode='probe',bind_ip=inventory[1]['ip'],duration_s=30.,
                 target_ip=inventory[7]['ip'],target_port=49665))
            report['probe']=remote[1].finish(probe)
            report['load']=remote[3].finish(load)
            report['servers']={str(port):remote[port].finish(handle) for port,handle in servers.items()}
            report['after']=snapshot(connection,prompt,commands)
            report['complete']=True
        finally:
            for endpoint in remote.values():
                endpoint.close()
            connection.sock.close()
            (a.out/('trial_%03d.json'%spec['ordinal'])).write_text(json.dumps(report,indent=2),encoding='utf-8')
        completed.append(spec['ordinal'])
        print(json.dumps({'switch_trials_complete':len(completed),'planned':len(trials),'block':spec['block'],
                          'path':spec['path'],'load_mbps':spec['wire_mbps'],'probes':report['probe']['result']['sent']}),flush=True)
    password=None
    (a.out/'completion.json').write_text(json.dumps(dict(complete=len(completed)==len(trials),trials=len(completed),utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')

if __name__=='__main__':
    main()
