"""Resume an interrupted formal campaign, repeating its whole incomplete block.

The parent directory stays immutable. No policy, workload, accounting rule,
randomized order, or timing is changed. Technical interruptions are retained
and whole-block replacement is decided before observing replacement outcomes.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from src.run_r2_physical import PORTS,load_services,source_hashes,jobs,nodes
from src.run_physical_trials import run_trial,dump
from src.calibrate_power_thermal import PowerSampler

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--aggregate',type=Path,default=Path('results/r2/workload_application_16x4.npz'))
    p.add_argument('--original',type=Path,default=Path('results/revision/physical/portable_model_20260905'))
    p.add_argument('--added',type=Path,default=Path('results/r2/trained_services_20261105'))
    p.add_argument('--redundant-power',action='store_true',help='Retain two staggered read-only PSE streams per trial')
    p.add_argument('--recover-power',action='store_true',help='Log short-timeout reconnects while enforcing continuous observed support')
    p.add_argument('--isolated-power',action='store_true',help='Collect PSE observations in a separate controller process')
    p.add_argument('--monitor-management',action='store_true',help='Retain two-endpoint 1-Hz ICMP diagnostics throughout replacement blocks')
    a=p.parse_args()
    if a.recover_power and not a.redundant_power:
        p.error('--recover-power requires --redundant-power')
    if a.isolated_power and not a.recover_power:
        p.error('--isolated-power requires --recover-power')
    parent=a.parent.resolve()
    out=a.out.resolve()
    if out==parent or out.is_relative_to(parent) or parent.is_relative_to(out):
        raise ValueError('Use distinct, non-overlapping campaign directories')
    status=json.loads((parent/'campaign_status.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text())
    if status['complete'] or not status['power_sampler']['thread_stopped']:
        raise ValueError('Parent must be stopped and incomplete')
    frozen=manifest['source_sha256']
    if source_hashes()!=frozen:
        raise ValueError('Execution sources changed. This is not a same-protocol recovery')
    inputs={'inventory':a.inventory,'service_calibration':a.calibration/'summary.json',
        'mlp':a.original/'model.json','cnn':a.added/'cnn/model.json','forest':a.added/'forest/model.json',
        'aggregate':a.aggregate,'forecast':Path('results/revision/forecast_prequential_20260905/forecast_prequential.npz'),
        'power':Path('results/revision/physical/power_thermal_20260905/analysis.json')}
    if {k:digest(v) for k,v in inputs.items()}!=manifest['input_sha256']:
        raise ValueError('A frozen input changed')
    completed=set(status['completed'])
    retained=[]
    first_incomplete=None
    for block in range(10):
        planned=[r['ordinal'] for r in manifest['trials'] if r['block']==block]
        if set(planned)<=completed and first_incomplete is None:
            retained.extend(planned)
        else:
            first_incomplete=block if first_incomplete is None else first_incomplete
    if first_incomplete is None:
        raise ValueError('No incomplete block found')
    for ordinal in retained:
        trial=json.loads((parent/('trial_%03d'%ordinal)/'summary.json').read_text())
        if not trial['complete'] or trial['error'] or trial['parity_failures']:
            raise ValueError('Retained trial is not complete')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','calibration_parameters.json','transport_reuse.json']:
        shutil.copy2(parent/name,out/name)
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    shutil.copytree(parent/'isolated_transport',out/'isolated_transport')
    inherited=out/'inherited_acquisition'
    inherited.mkdir()
    for name in ['power.jsonl','power_sessions.jsonl','power_failure.json','campaign_status.json']:
        if (parent/name).exists():
            shutil.copy2(parent/name,inherited/name)
    history=[]
    historical_errors=list(status['power_sampler']['errors'])
    if (parent/'resumption.json').exists():
        previous=json.loads((parent/'resumption.json').read_text())
        historical_errors=previous['historical_power_errors']+historical_errors
        prior=out/'recovery_history'
        prior.mkdir()
        if (parent/'recovery_history').exists():
            shutil.copytree(parent/'recovery_history',prior,dirs_exist_ok=True)
        record=prior/parent.name
        record.mkdir()
        for name in ['resumption.json','campaign_status.json']:
            shutil.copy2(parent/name,record/name)
        shutil.copy2(parent/'source_snapshot/resume_r2_physical.py',record/'resume_r2_physical.py')
        for source_name in previous.get('additional_acquisition_sources',{}):
            shutil.copy2(parent/'source_snapshot'/source_name,record/source_name)
        history=[dict(path=p.relative_to(out).as_posix(),sha256=digest(p)) for p in sorted(prior.rglob('*')) if p.is_file()]
    copied=[]
    for ordinal in retained:
        name='trial_%03d'%ordinal
        shutil.copytree(parent/name,out/name)
        for source in sorted((parent/name).glob('*')):
            if source.is_file():
                value=digest(source)
                if digest(out/name/source.name)!=value:
                    raise AssertionError('Copied trial hash mismatch')
                copied.append(dict(path=name+'/'+source.name,sha256=value))
    self_path=Path(__file__)
    shutil.copy2(self_path,out/'source_snapshot'/self_path.name)
    recovery=dict(registered_utc=datetime.now(timezone.utc).isoformat(),parent_directory=parent.name,
        parent_manifest_sha256=digest(parent/'manifest.json'),parent_status_sha256=digest(parent/'campaign_status.json'),
        parent_power_sha256=digest(parent/'power.jsonl'),retained_complete_blocks=list(range(first_incomplete)),
        retained_ordinals=retained,excluded_parent_completed_ordinals=sorted(completed-set(retained)),
        first_repeated_block=first_incomplete,reason='Read-only power acquisition interruption. Entire incomplete matched block repeated.',
        rule='Registered before replacement outcomes. Original complete blocks are byte-preserved. Every attempted record remains in the parent archive.',
        frozen_execution_unchanged=True,frozen_inputs_unchanged=True,
        resumption_source_sha256=digest(self_path),copied_records=copied,
        historical_power_errors=historical_errors,recovery_history=history)
    if a.redundant_power:
        acquisition_source=Path(__file__).with_name('redundant_power_sampler.py')
        shutil.copy2(acquisition_source,out/'source_snapshot'/acquisition_source.name)
        recovery['additional_acquisition_sources']={acquisition_source.name:digest(acquisition_source)}
        recovery['instrumentation_amendment']=dict(
            reason='Repeated read-only management-stream timeouts interrupted different network-condition policies.',
            first_block=first_incomplete,scope='Observation redundancy only. Algorithm, job order, budget, model, timing and board cohort are unchanged.',
            acquisition='Two staggered independent 1-Hz streams per trial. Retain both streams and all failures. Renew only between trials.',
            inference='Keep all ten matched blocks. Require observed support for each energy window. No interpolation across an unsupported gap.')
        if a.recover_power:
            acquisition_source=Path(__file__).with_name('recovering_power_sampler.py')
            shutil.copy2(acquisition_source,out/'source_snapshot'/acquisition_source.name)
            recovery['additional_acquisition_sources'][acquisition_source.name]=digest(acquisition_source)
            recovery['instrumentation_amendment']['acquisition']='Two staggered 1-Hz read-only streams. Retain every failed query and reauthenticate after a 1-s query timeout. Stop if observed support is absent for more than 2.5 s. Fresh pairs between trials.'
        if a.isolated_power:
            acquisition_source=Path(__file__).with_name('power_collection_process.py')
            shutil.copy2(acquisition_source,out/'source_snapshot'/acquisition_source.name)
            recovery['additional_acquisition_sources'][acquisition_source.name]=digest(acquisition_source)
            recovery['instrumentation_amendment']['process_isolation']='Dedicated collector process on the same Windows host, using its common monotonic clock. No algorithm, workload or energy-accounting changes.'
    if a.monitor_management:
        acquisition_source=Path(__file__).with_name('management_path_monitor.py')
        shutil.copy2(acquisition_source,out/'source_snapshot'/acquisition_source.name)
        recovery.setdefault('additional_acquisition_sources',{})[acquisition_source.name]=digest(acquisition_source)
        recovery['reachability_monitor']='Controller and port 1 each emit one ICMP echo per second. All replies and failures are retained. Monitor outputs never enter policy decisions or power integration.'
    dump(out/'resumption.json',recovery)
    print(json.dumps({k:recovery[k] for k in ['retained_complete_blocks','first_repeated_block','excluded_parent_completed_ordinals']}),flush=True)
    password=getpass.getpass('Switch password (hidden): ')
    rows=[r for r in json.loads(a.inventory.read_text())['ports'] if r['port'] in PORTS]
    key=os.environ['EDGE_SSH_KEY']
    model,examples=load_services(a.original,a.added)
    params=json.loads((out/'calibration_parameters.json').read_text())
    datasets=dict(service=params['service'],transport=params['transport'],power=params['power'],
                  aggregate=dict(np.load(a.aggregate)),forecast=dict(np.load(inputs['forecast'])))
    acquisition=out/'resumed_acquisition'
    acquisition.mkdir()
    sampler=None
    monitor=None
    accepted=retained[:]
    try:
        if a.monitor_management:
            from src.management_path_monitor import ManagementPathMonitor
            monitor=ManagementPathMonitor(out/'management_path',next(r for r in rows if r['port']==1),key,os.environ['EDGE_SWITCH_HOST'])
        if a.isolated_power:
            from src.power_collection_process import ProcessPowerSampler
            sampler=ProcessPowerSampler(acquisition,password)
        elif a.redundant_power:
            from src.redundant_power_sampler import RedundantPowerSampler
            sampler=RedundantPowerSampler(acquisition,password,recover=a.recover_power)
        else:
            sampler=PowerSampler(acquisition,password)
        password=None
        for spec in manifest['trials']:
            if spec['ordinal'] in retained:
                continue
            if source_hashes()!=frozen:
                raise RuntimeError('Frozen execution source changed')
            for name,expected in recovery.get('additional_acquisition_sources',{}).items():
                if digest(Path(__file__).with_name(name))!=expected:
                    raise RuntimeError('Registered acquisition source changed')
            if spec['ordinal']!=(retained[-1]+1 if retained else 0):
                sampler.renew_between_trials()
            result=run_trial(spec,out/('trial_%03d'%spec['ordinal']),rows,key,model,
                dict(services=examples),datasets,sampler,job_generator=jobs,node_builder=nodes)
            accepted.append(spec['ordinal'])
            print(json.dumps(dict(event='r2_trial_complete',completed=len(accepted),total=140,
                block=spec['block'],scenario=spec['scenario'],workload=spec['workload'],method=spec['method'],
                successful=result['successful_inferences'],rejected=result['rejected_batches'])),flush=True)
    finally:
        password=None
        sampling=sampler.close() if sampler else None
        monitoring=monitor.close() if monitor else None
        # Concatenation preserves every observed value and the monotonic clock.
        # Historical gaps outside included trial windows are not interpolated.
        with (out/'power.jsonl').open('xb') as combined:
            for path in [inherited/'power.jsonl',acquisition/'power.jsonl']:
                if path.exists():
                    with path.open('rb') as handle:
                        shutil.copyfileobj(handle,combined)
        dump(out/'campaign_status.json',dict(complete=len(accepted)==140,completed=accepted,
            ended_utc=datetime.now(timezone.utc).isoformat(),power_sampler=sampling,
            inherited_power_sampler=status['power_sampler'],resumption='resumption.json',management_monitor=monitoring,
            historical_errors_scope='The interrupted parent block is excluded as a whole. Included-window continuity is checked separately.'))

if __name__=='__main__':
    main()
