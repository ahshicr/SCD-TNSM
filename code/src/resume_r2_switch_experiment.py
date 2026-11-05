"""Repeat an interrupted whole switch block without selecting its conditions.

Only connection establishment, before counter collection and traffic, gains
bounded retries. The frozen packet generator and trial parameters are reused.
"""
import argparse
from datetime import datetime,timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from src.run_r2_switch_experiment import Remote,snapshot,Connection


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect_before_trial(directory,ordinal,password):
    prompt=r'(?:^|[\r\n])\s*<[^\r\n>]+>\s*$'
    for attempt in range(3):
        start=time.monotonic()
        connection=None
        stage='tcp'
        record=dict(ordinal=ordinal,attempt=attempt,started_monotonic=start)
        try:
            connection=Connection(os.environ['EDGE_SWITCH_HOST'])
            stage='username_prompt'
            if not re.search(r'(?:login|username)\s*:',connection.read(r'(?:login|username)\s*:',20),re.I):
                raise RuntimeError('Username prompt unavailable')
            connection.send(os.environ['EDGE_SWITCH_USERNAME'])
            stage='password_prompt'
            if not re.search(r'password\s*:',connection.read(r'password\s*:',20),re.I):
                raise RuntimeError('Password prompt unavailable')
            connection.send(password)
            stage='command_prompt'
            if not re.search(prompt,connection.read(prompt,20)):
                raise RuntimeError('Authenticated command prompt unavailable')
            record.update(complete=True,stage=stage)
            return connection,prompt
        except (OSError,RuntimeError) as error:
            record.update(complete=False,stage=stage,error_type=type(error).__name__)
            if connection is not None:
                record['transport_diagnostic']=getattr(connection,'last_read',{})
                connection.sock.close()
            if attempt==2:
                raise
        finally:
            record['ended_monotonic']=time.monotonic()
            # Authentication replies and values are deliberately not recorded.
            with (directory/'management_connections.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
        time.sleep(2.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    protocol=read(args.parent/'protocol.json')
    trials=protocol['trials']
    if len(trials)!=60 or (args.parent/'completion.json').is_file():
        raise ValueError('Resume only the preserved incomplete original campaign')
    completed=[]
    for spec in trials:
        path=args.parent/('trial_%03d.json'%spec['ordinal'])
        if not path.is_file():
            break
        record=read(path)
        if record['spec']!=spec or not record['complete']:
            break
        completed.append(spec['ordinal'])
    retained=len(completed)//6*6
    if retained==0 or retained==60:
        raise ValueError('Expected an interrupted campaign with whole completed blocks')
    for name,digest in protocol['source_sha256'].items():
        if sha(args.parent/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('Frozen acquisition source changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.parent/'protocol.json',args.out/'protocol.json')
    shutil.copytree(args.parent/'source_snapshot',args.out/'source_snapshot')
    copied={}
    for ordinal in range(retained):
        name='trial_%03d.json'%ordinal
        shutil.copy2(args.parent/name,args.out/name)
        copied[name]=sha(args.parent/name)
        if sha(args.out/name)!=copied[name]:
            raise ValueError('Retained trial copy changed')
    addition=args.out/'resumption_sources'
    addition.mkdir()
    shutil.copy2(Path(__file__),addition/Path(__file__).name)
    recovery=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        parent_directory=args.parent.name,retained_complete_blocks=retained//6,
        first_repeated_block=retained//6,retained_trial_sha256=copied,
        excluded_parent_complete_ordinals=completed[retained:],
        parent_protocol_sha256=sha(args.parent/'protocol.json'),
        additional_source_sha256={Path(__file__).name:sha(Path(__file__))},
        cause='Original process stopped at connection establishment before the next trial. No new traffic or counters from that trial were recorded.',
        rule='Preserve preceding complete six-condition blocks. Repeat the entire incomplete block and every remaining block in its original registered order.',
        amendment='At most three connection attempts, with 20-second prompt waits and two-second backoff, occur before any trial traffic or counter window. No within-trial retry or missing counter imputation.')
    (args.out/'resumption.json').write_text(json.dumps(recovery,indent=2),encoding='utf-8')
    inventory={r['port']:r for r in read(args.inventory)['ports']}
    key=os.environ['EDGE_SSH_KEY']
    commands=['display interface GigabitEthernet1/0/'+str(p) for p in [1,3,7,23]]
    password=getpass.getpass('Switch password (hidden): ')
    completed=list(range(retained))
    for spec in trials[retained:]:
        for name,digest in protocol['source_sha256'].items():
            if sha(Path(__file__).with_name(name))!=digest:
                raise RuntimeError('Frozen acquisition source changed')
        if sha(Path(__file__))!=recovery['additional_source_sha256'][Path(__file__).name]:
            raise RuntimeError('Resumption source changed')
        connection,prompt=connect_before_trial(args.out,spec['ordinal'],password)
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
            (args.out/('trial_%03d.json'%spec['ordinal'])).write_text(json.dumps(report,indent=2),encoding='utf-8')
        completed.append(spec['ordinal'])
        print(json.dumps(dict(switch_trials_complete=len(completed),planned=60,block=spec['block'],
            path=spec['path'],load_mbps=spec['wire_mbps'],probes=report['probe']['result']['sent'])),flush=True)
    password=None
    (args.out/'completion.json').write_text(json.dumps(dict(complete=completed==list(range(60)),
        trials=len(completed),utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')


if __name__=='__main__':
    main()
