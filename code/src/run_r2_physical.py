"""Complete R2 matched-constraint and mixed-service live-board campaign."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from src.board_inference_client import BoardClient
from src.physical_policy import LiveNode
from src.physical_trial_design import jobs_for_trial
from src.run_physical_trials import run_trial, dump
from src.calibrate_power_thermal import PowerSampler
from src.calibrate_r2_services import PORTS, load_services

METER_PORTS = [1,3,4,5,7,8,9]

def design():
    trials=[]
    for block in range(10):
        rng=random.Random(20261105+block)
        conditions=[('nominal','mlp'),('energy_pressure','mlp'),('network_pressure','mlp'),('nominal','mixed')]
        rng.shuffle(conditions)
        for scenario,workload in conditions:
            methods=['compute_only','dpp_matched','scd']
            if scenario=='network_pressure':
                methods+=['scd_no_link','scd_independent_network']
            rng.shuffle(methods)
            for method in methods:
                trials.append(dict(ordinal=len(trials),block=block,cohort='r2_fleet14',ports=PORTS,
                    scenario=scenario,workload=workload,method=method,constraint_mode='matched',
                    duration_s=60.,drain_s=15.,idle_s=20.,jobs=2880,
                    budget_factor=.06 if scenario=='energy_pressure' else .5,v_joules=1.,energy_margin=1.1,
                    trace_start=2880+1440*(block%12)+300,trace_minutes=60,job_seed=20261105+1009*block,
                    hardware_guard_c=65.,max_inflight_per_board=4))
    return trials

def jobs(spec,aggregate,forecast):
    result=jobs_for_trial(spec,aggregate,forecast)
    assignment=np.tile(np.arange(3),len(result)//3+1)[:len(result)]
    np.random.default_rng(spec['job_seed']+77).shuffle(assignment)
    for i,job in enumerate(result):
        job['service_id']=['mlp','cnn','forest'][int(assignment[i])] if spec['workload']=='mixed' else 'mlp'
    return result

def nodes(cohort,service,transport,power,initial):
    calibrated={r['port']:r for r in power['boards'] if r.get('achieved_occupancy_fit')}
    result=[]
    for port in PORTS:
        reference=port if port in calibrated else (5 if port in [1,3,4,19] else 9 if port in [10,17,21,22] else 7 if port==23 else None)
        if reference is not None:
            fit=calibrated[reference]['achieved_occupancy_fit']
            idle,incremental=fit['idle_intercept_w'],fit['active_increment_w']
            provenance='R1 active-CPU calibration on port %s. Task-charge coefficient reused for every policy and service.'%reference
        else:
            idle,incremental=4.,3.
            provenance='RDK declared unmetered coefficient, not individual power measurement'
        observed=initial[str(port)]['telemetry']['cpu_c']
        model=service[str(port)]
        link=transport[str(port)]
        node=LiveNode(port,model['mlp'],idle,incremental,100. if port in [7,8] else 1000.,
            link['fixed_s'],link['seconds_per_mb'],provenance,observed,
            healthy=observed is not None and observed<65.,service_by_workload=model)
        result.append(node)
    return result

def transport_calibration(rows,key,model,examples,out):
    out.mkdir(exist_ok=False)
    jobs=[(size,repeat) for size in [264,65536,1048576,4194304] for repeat in range(10)]
    random.Random(20261105).shuffle(jobs)
    result={}
    for row in rows:
        client=BoardClient(row,key,model)
        records=[]
        try:
            with (out/('port_%02d.jsonl'%row['port'])).open('x',encoding='utf-8') as f:
                for ordinal,(size,repeat) in enumerate(jobs):
                    data=client.infer(ordinal,examples['forest']['features'][repeat:repeat+1],size,'forest')
                    if data['status']!='ok' or data['predictions'][0] != examples['forest']['expected'][repeat]:
                        raise RuntimeError('Transport activation failed output parity')
                    data.update(size=size,repetition=repeat)
                    f.write(json.dumps(data)+'\n')
                    f.flush()
                    records.append(data)
            x=np.array([r['size']/1e6 for r in records])
            y=np.array([r['controller_latency_s']-(r['compute_end']-r['compute_start']) for r in records])
            from scipy.optimize import nnls
            beta,_=nnls(np.column_stack([np.ones(len(x)),x]),y)
            result[str(row['port'])]=dict(fixed_s=float(beta[0]),seconds_per_mb=float(beta[1]),groups=len(records))
        finally:
            cleanup=client.close()
            if cleanup['errors'] or not cleanup['directory_removed']:
                raise RuntimeError('Unconfirmed transport-worker cleanup')
        print(json.dumps({'event':'transport_calibrated','port':row['port'],'groups':len(records)}),flush=True)
    (out/'models.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def source_hashes():
    names=['run_r2_physical.py','run_physical_trials.py','physical_trial_design.py','physical_policy.py',
           'scheduler_core.py','board_inference_client.py','board_inference_worker.py','board_pipeline.py',
           'portable_inference.py','r2_portable_models.py','calibrate_r2_services.py','calibrate_power_thermal.py',
           'poe_sampling_audit.py','switch_transport.py']
    return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--simulation-complete',type=Path,required=True)
    p.add_argument('--aggregate',type=Path,default=Path('results/r2/workload_application_16x4.npz'))
    p.add_argument('--reuse-transport',type=Path)
    p.add_argument('--original',type=Path,default=Path('results/revision/physical/portable_model_20260905'))
    p.add_argument('--added',type=Path,default=Path('results/r2/trained_services_20261105'))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    trials=design()
    frozen=source_hashes()
    input_paths={'inventory':a.inventory,'service_calibration':a.calibration/'summary.json',
        'mlp':a.original/'model.json','cnn':a.added/'cnn/model.json','forest':a.added/'forest/model.json',
        'aggregate':a.aggregate,
        'forecast':Path('results/revision/forecast_prequential_20260905/forecast_prequential.npz'),
        'power':Path('results/revision/physical/power_thermal_20260905/analysis.json')}
    manifest=dict(schema='tnsm.r2.live.v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        trials=trials,planned_trials=len(trials),source_sha256=frozen,
        input_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in input_paths.items()},
        matched_constraints='All policies use identical device and tenant joule buckets, request token caps, safety guards, and inflight bounds. DPP differs from SCD only by deadline filtering.',
        primary_metrics=['offered on-time inference success','request-weighted mean and P95 response','rejection ratio'],
        power_scope='Observed PSE subtotal on ports 1,3,4,5,7,8,9. This is not total fleet energy. Compare only this fixed subtotal, also with measured pre-trial idle removed.',
        metered_ports=METER_PORTS,missing_energy_ports=sorted(set(PORTS)-set(METER_PORTS)),
        statistics='Ten paired randomized blocks. All prespecified contrasts and outcomes retained. Holm adjustment across primary contrasts.',
        services=['trained demand MLP','trained 8-channel image CNN','trained 64-tree image classifier'],
        execution='No simultaneous simulation, calibration or external traffic generation during scheduler trials',
        tenant_definition='Same stable application-identity modulo-four grouping as the request-level simulator. Trigger category is not a tenant.',
        protocol_amendment='The original physical generator grouped trigger categories. Before formal R2 inference, replace those counts with application-coherent counts. All earlier R2 commissioning trials are retained and excluded as a whole, without selecting performance outcomes.',
        artifact_privacy='Credentials never persisted. Endpoint IPs and host keys are private inventory, not public artifacts.')
    dump(a.out/'manifest.json',manifest)
    snapshot=a.out/'source_snapshot'
    snapshot.mkdir()
    for name in frozen:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    password=getpass.getpass('Switch password (hidden, retained only in memory): ')
    while not a.simulation_complete.exists() or not (a.calibration/'summary.json').exists():
        time.sleep(5)
    calibration=json.loads((a.calibration/'summary.json').read_text(encoding='utf-8'))
    if not calibration['complete']:
        raise RuntimeError('Calibration incomplete')
    if source_hashes()!=frozen:
        raise RuntimeError('Frozen execution source changed')
    rows=[r for r in json.loads(a.inventory.read_text(encoding='utf-8'))['ports'] if r['port'] in PORTS]
    key=os.environ['EDGE_SSH_KEY']
    model,examples=load_services(a.original,a.added)
    service={str(r['port']):r['service_models'] for r in calibration['boards']}
    if a.reuse_transport:
        transport=json.loads((a.reuse_transport/'models.json').read_text(encoding='utf-8'))
        if set(transport)!=set(map(str,PORTS)) or any(r['groups']!=40 for r in transport.values()):
            raise ValueError('Reused transport calibration is incomplete')
        shutil.copytree(a.reuse_transport,a.out/'isolated_transport')
        dump(a.out/'transport_reuse.json',dict(reason='Complete isolated transport calibration is independent of tenant grouping',
            model_sha256=hashlib.sha256((a.reuse_transport/'models.json').read_bytes()).hexdigest()))
    else:
        transport=transport_calibration(rows,key,model,examples,a.out/'isolated_transport')
    base=Path('results/revision')
    datasets=dict(service=service,transport=transport,
        power=json.loads((base/'physical/power_thermal_20260905/analysis.json').read_text(encoding='utf-8')),
        aggregate=dict(np.load(a.aggregate)),
        forecast=dict(np.load(base/'forecast_prequential_20260905/forecast_prequential.npz')))
    dump(a.out/'calibration_parameters.json',dict(service=service,transport=transport,power=datasets['power']))
    sampler=None
    completed=[]
    try:
        sampler=PowerSampler(a.out,password)
        password=None
        for spec in trials:
            if source_hashes()!=frozen:
                raise RuntimeError('Frozen execution source changed')
            if spec['ordinal']:
                sampler.renew_between_trials()
            result=run_trial(spec,a.out/('trial_%03d'%spec['ordinal']),rows,key,model,dict(services=examples),datasets,sampler,
                             job_generator=jobs,node_builder=nodes)
            completed.append(spec['ordinal'])
            print(json.dumps(dict(event='r2_trial_complete',completed=len(completed),total=len(trials),
                block=spec['block'],scenario=spec['scenario'],workload=spec['workload'],method=spec['method'],
                successful=result['successful_inferences'],rejected=result['rejected_batches'])),flush=True)
    finally:
        password=None
        dump(a.out/'campaign_status.json',dict(complete=len(completed)==len(trials),completed=completed,
             ended_utc=datetime.now(timezone.utc).isoformat(),power_sampler=sampler.close() if sampler else None))

if __name__=='__main__':
    main()
