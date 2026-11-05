"""Audit all formal R2 live trials and generate tables from their raw records.

The statistical unit is a randomized matched block. This analysis never pools
requests, calibration repetitions, ports, or one-second samples as replicates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
from src.physical_trace_audit import audit_trace
from src.summarize_physical_trials import weighted_quantile, integrate_observed
from src.audit_physical_power_sampling import interval_support
from src.r2_evidence_macros import write_macros

PRIMARY=['offered_on_time_success','mean_latency_ms','p95_latency_ms','rejection_fraction']
SECONDARY=['measured_pse_subtotal_j','idle_adjusted_pse_subtotal_j','idle_pse_subtotal_w',
           'predicted_incremental_charge_j','jain_tenant_success','minimum_tenant_success',
           'peak_observed_cpu_c','mean_decision_ms','p95_dispatch_path_ms',
           'guard_batches','censored_batches','completed_fraction']
LABELS={'scd':'SCD','dpp_matched':'Matched DPP','compute_only':'Compute-first',
        'scd_no_link':'No network penalty','scd_independent_network':'Independent link state'}
CONDITIONS=[('nominal','mlp','MLP nominal'),('energy_pressure','mlp','MLP energy pressure'),
            ('network_pressure','mlp','MLP network pressure'),('nominal','mixed','Mixed nominal')]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def failure_intervals(record):
    """Flatten single-connection and redundant-collector failure evidence."""
    if isinstance(record,dict):
        if 'query_start' in record and 'query_end' in record:
            yield float(record['query_start']),float(record['query_end'])
        elif 'latest_midpoint' in record and 'monotonic' in record:
            yield float(record['latest_midpoint']),float(record['monotonic'])
        else:
            for value in record.values():
                yield from failure_intervals(value)
    elif isinstance(record,list):
        for value in record:
            yield from failure_intervals(value)

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def origin_verifier(directory):
    """Accept only unchanged bytes or an explicitly registered public redaction."""
    root=directory.resolve()
    while root.parent!=root and not (root/'PUBLIC_PROVENANCE.json').is_file():
        root=root.parent
    records={}
    if (root/'PUBLIC_PROVENANCE.json').is_file():
        rows=read(root/'PUBLIC_PROVENANCE.json')['files']
        records={r['path']:r for r in rows}
        if len(records)!=len(rows):
            raise ValueError('Duplicate public provenance paths')
    def verify(path,expected):
        actual=sha(path)
        if actual==expected:
            return
        relative=path.resolve().relative_to(root).as_posix() if path.resolve().is_relative_to(root) else None
        record=records.get(relative)
        if not record or record['original_sha256']!=expected or record['public_sha256']!=actual:
            raise ValueError('Unverified origin hash for '+path.name)
        allowed={'deployment_redaction_or_portable_path_serialization','historical_connection_defaults_to_environment'}
        if record['transformation'] not in allowed:
            raise ValueError('Unregistered provenance transformation for '+path.name)
    return verify

def interval(values):
    values=np.asarray(values,float)
    if len(values)!=10:
        raise ValueError('All ten randomized blocks are required')
    n=int(np.isfinite(values).sum())
    if n!=10:
        # Keep undefined outcomes visible instead of selecting finite trials.
        return float('nan'),float('nan'),n
    return float(values.mean()),float(t.ppf(.975,9)*values.std(ddof=1)/np.sqrt(10)),n

def probability(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all() or np.all(values==0):
        return 1.
    return 0. if np.std(values)==0 else float(ttest_1samp(values,0).pvalue)

def holm(rows,field='p_unadjusted',output='p_holm_primary'):
    previous=0.
    for rank,index in enumerate(sorted(range(len(rows)),key=lambda k:rows[k][field])):
        previous=max(previous,min(1.,(len(rows)-rank)*rows[index][field]))
        rows[index][output]=previous

def analyze_trial(directory,spec,power_times,power_values):
    summary=read(directory/'summary.json')
    protocol=read(directory/'protocol.json')
    params=read(directory/'model_parameters.json')
    jobs=read(directory/'offered_jobs.json')
    ready=read(directory/'worker_ready.json')
    initial=read(directory/'initial_status.json')
    if not summary['complete'] or summary['error'] or summary['parity_failures']:
        raise ValueError('Incomplete or invalid formal trial')
    if set(ready)!=set(map(str,spec['ports'])) or set(initial)!=set(ready):
        raise ValueError('The declared board cohort was not initialized completely')
    if any('error' in r for r in ready.values()) or any('error' in r for r in initial.values()):
        raise ValueError('A declared endpoint did not initialize')
    if any(c.get('errors') or not c.get('stop_acknowledged') or not c.get('directory_removed')
           or not c.get('pipeline_threads_stopped',True) for c in summary['cleanup'].values()):
        raise ValueError('Incomplete worker cleanup')
    if not params['energy_guard'] or not params['request_guard'] or params['constraint_mode']!='matched':
        raise ValueError('Resource constraints are not common to all policies')
    if [n['port'] for n in params['nodes']]!=spec['ports'] or summary['ports']!=spec['ports']:
        raise ValueError('Endpoint order/cohort changed')
    digest=hashlib.sha256(json.dumps(jobs,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if protocol['offered_sha256']!=digest:
        raise ValueError('Offered workload hash mismatch')
    events=[]
    with (directory/'events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line))
    audit=audit_trace(events,params,spec)
    dispatched,results,rejected={}, {},set()
    decision_times,dispatch_times=[],[]
    ports=Counter()
    temperatures={p:[] for p in spec['ports']}
    for event in events:
        kind=event['event']
        if kind=='decision':
            identity=event['job']['id']
            if identity in dispatched or identity in rejected:
                raise ValueError('Duplicate decision')
            decision_times.append(event['decision_s'])
            dispatch_times.append(event['dispatch_path_s'])
            choice=event['decision']['selected']
            if choice is None or 'submission_error' in event:
                rejected.add(identity)
            else:
                c=event['decision']['candidates'][choice]
                if not c['eligible'] or c['predicted_energy_j']>min(c['device_credit_j'],c['tenant_credit_j'])+1e-8:
                    raise ValueError('Dispatched job violates matched energy eligibility')
                if c['inflight']>=4 or c['observed_temperature_c'] is None or c['observed_temperature_c']>=65:
                    raise ValueError('Common operational guard violated')
                dispatched[identity]=c
                ports[c['port']]+=event['job']['batch']
        elif kind=='rejected':
            identity=event['job']['id']
            if identity in dispatched or identity in rejected:
                raise ValueError('Duplicate rejection')
            rejected.add(identity)
        elif kind=='result':
            identity=event['job']['id']
            if identity in results:
                raise ValueError('Duplicate result')
            result=event['result']
            results[identity]=result
            for endpoint in ['before','after']:
                value=result.get(endpoint,{}).get('cpu_c')
                if value is not None and np.isfinite(value):
                    temperatures[result['port']].append(value)
        elif kind=='state':
            for p,value in zip(spec['ports'],event['temperature_c']):
                if value is not None and np.isfinite(value):
                    temperatures[p].append(value)
    offered={j['id'] for j in jobs}
    if len(offered)!=2880 or (set(dispatched)&rejected) or set(dispatched)|rejected!=offered:
        raise ValueError('Offered-job conservation failed')
    if not set(results)<=set(dispatched):
        raise ValueError('Unassigned result')
    censored=set(summary['outstanding_at_fixed_drain'])
    if set(dispatched)-set(results)!=censored:
        raise ValueError('Unreported drain censoring')
    byservice={name:dict(offered=0,on_time=0,completed=0,rejected=0,latencies=[],weights=[]) for name in ['mlp','cnn','forest']}
    tenant_offered=np.zeros(4)
    tenant_success=np.zeros(4)
    latencies,weights=[],[]
    successes=on_time=total_requests=0
    for job in jobs:
        weight=job['batch']
        total_requests+=weight
        tenant_offered[job['tenant']]+=weight
        service=byservice[job['service_id']]
        service['offered']+=weight
        service['rejected']+=weight*int(job['id'] in rejected)
        result=results.get(job['id'])
        if result and result['batch_atomic_success']:
            if not result['parity_ok'] or len(result['predictions'])!=weight or not np.isfinite(result['release_to_result_s']):
                raise ValueError('Invalid completed batch')
            latency=result['release_to_result_s']
            ok=latency<=job['deadline_s']
            on_time+=weight*int(ok)
            successes+=weight
            tenant_success[job['tenant']]+=weight*int(ok)
            latencies.append(latency)
            weights.append(weight)
            service['on_time']+=weight*int(ok)
            service['completed']+=weight
            service['latencies'].append(latency)
            service['weights'].append(weight)
    if successes!=summary['successful_inferences'] or total_requests!=protocol['offered_inferences']:
        raise ValueError('Completion or offered inference total mismatch')
    active=tenant_offered>0
    fractions=tenant_success[active]/tenant_offered[active]
    no_service=bool(not tenant_success.sum())
    jain=0. if no_service else float(fractions.sum()**2/(len(fractions)*np.dot(fractions,fractions)))
    row=dict(ordinal=spec['ordinal'],block=spec['block'],scenario=spec['scenario'],workload=spec['workload'],method=spec['method'],
        offered_batches=len(jobs),offered_requests=total_requests,admitted_requests=sum(j['batch'] for j in jobs if j['id'] in dispatched),
        completed_requests=successes,on_time_requests=on_time,offered_on_time_success=on_time/total_requests,
        completed_fraction=successes/total_requests,rejection_fraction=sum(j['batch'] for j in jobs if j['id'] in rejected)/total_requests,
        mean_latency_ms=float(np.average(latencies,weights=weights)*1000) if successes else None,
        p95_latency_ms=weighted_quantile(latencies,weights,.95)*1000 if successes else None,
        mean_decision_ms=float(np.mean(decision_times)*1000),p95_dispatch_path_ms=float(np.quantile(dispatch_times,.95)*1000),
        guard_batches=summary['guarded_batches'],censored_batches=len(censored),failed_batches=summary['failed_batches'],
        predicted_incremental_charge_j=sum(c['predicted_energy_j'] for c in dispatched.values()),
        peak_observed_cpu_c=max((max(v) for v in temperatures.values() if v),default=None),
        jain_tenant_success=jain,minimum_tenant_success=float(fractions.min()),no_service=no_service,
        measured_pse_subtotal_j=None,idle_adjusted_pse_subtotal_j=None,idle_pse_subtotal_w=None)
    begin,end=summary['arrival_start_monotonic'],summary['measurement_end_monotonic']
    if abs(end-begin-75)>1e-5:
        raise ValueError('Unequal energy windows')
    support={name:interval_support(power_times,a,b) for name,a,b in [('active',begin,end),('idle',begin-15,begin)]}
    total_power=power_values.sum(axis=1)
    try:
        row['measured_pse_subtotal_j']=integrate_observed(power_times,total_power,begin,end)
        row['idle_pse_subtotal_w']=integrate_observed(power_times,total_power,begin-15,begin)/15
        row['idle_adjusted_pse_subtotal_j']=row['measured_pse_subtotal_j']-75*row['idle_pse_subtotal_w']
        support['energy_available']=True
    except ValueError as exc:
        # Never bridge an unobserved internal gap to manufacture an energy value.
        row.update(measured_pse_subtotal_j=None,idle_pse_subtotal_w=None,idle_adjusted_pse_subtotal_j=None)
        support.update(energy_available=False,reason=str(exc))
    common={k:row[k] for k in ['ordinal','block','scenario','workload','method']}
    port_rows=[dict(**common,port=p,assigned_requests=ports[p],share_of_admitted=ports[p]/max(1,row['admitted_requests']),
                   peak_cpu_c=max(temperatures[p],default=None)) for p in spec['ports']]
    service_rows=[]
    for name,service in byservice.items():
        if not service['offered']:
            continue
        s=dict(**common,service=name,offered_requests=service['offered'],on_time_requests=service['on_time'],
               completed_requests=service['completed'],offered_on_time_success=service['on_time']/service['offered'],
               rejection_fraction=service['rejected']/service['offered'])
        s['mean_latency_ms']=float(np.average(service['latencies'],weights=service['weights'])*1000) if service['completed'] else None
        s['p95_latency_ms']=weighted_quantile(service['latencies'],service['weights'],.95)*1000 if service['completed'] else None
        service_rows.append(s)
    return row,port_rows,service_rows,dict(**common,**audit),support,protocol['offered_sha256'],params['budget_j_per_minute']

def cell(mean,half,scale=1,digits=2):
    if not np.isfinite(mean):
        return r'\textemdash'
    displayed=f'{mean*scale:.{digits}f}'
    if float(displayed)==0:
        displayed=displayed.lstrip('-')
    return '$'+displayed+r'\pm'+f'{half*scale:.{digits}f}'+'$'

def probability_cell(value):
    if not np.isfinite(value):
        return r'\textemdash'
    text=f'{value:.3g}'
    if 'e' in text:
        mantissa,exponent=text.split('e')
        text=mantissa+r'\times10^{'+str(int(exponent))+'}'
    return '$'+text+'$'

def summarize(directory,tex,out=None):
    manifest=read(directory/'manifest.json')
    campaign=read(directory/'campaign_status.json')
    verify_origin=origin_verifier(directory)
    if not campaign['complete'] or campaign['completed']!=list(range(140)) or manifest['planned_trials']!=140:
        raise ValueError('All 140 formal trials must finish before analysis')
    if campaign['power_sampler']['errors'] or not campaign['power_sampler']['thread_stopped']:
        raise ValueError('Power collection failed')
    for name,digest in manifest['source_sha256'].items():
        verify_origin(directory/'source_snapshot'/name,digest)
    recovery=None
    if (directory/'resumption.json').exists():
        recovery=read(directory/'resumption.json')
        verify_origin(directory/'source_snapshot/resume_r2_physical.py',recovery['resumption_source_sha256'])
        for name,expected in recovery.get('additional_acquisition_sources',{}).items():
            verify_origin(directory/'source_snapshot'/name,expected)
        for entry in recovery.get('recovery_history',[]):
            verify_origin(directory/entry['path'],entry['sha256'])
        retained=set(recovery['retained_ordinals'])
        for block in recovery['retained_complete_blocks']:
            if {r['ordinal'] for r in manifest['trials'] if r['block']==block}-retained:
                raise ValueError('A partial inherited block was retained')
        # Copy hashes are origin hashes. A public export may redact deployment
        # identifiers and records that transformation in PUBLIC_PROVENANCE.
        for entry in recovery['copied_records']:
            verify_origin(directory/entry['path'],entry['sha256'])
    out=out or directory/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    times,powers=[],[]
    with (directory/'power.jsonl').open(encoding='utf-8') as f:
        for line in f:
            sample=json.loads(line)
            values={p['port']:p['poe_output_w'] for p in sample['ports']}
            times.append((sample['query_start']+sample['query_end'])/2)
            powers.append([values.get(p) for p in manifest['metered_ports']])
    times,powers=np.asarray(times),np.asarray(powers,dtype=float)
    if not len(times) or np.any(np.diff(times)<=0):
        raise ValueError('Unordered/missing power samples')
    records,ports,services,certificates,supports=[],[],[],[],[]
    offered,budgets,hashes={},{},{}
    for spec in manifest['trials']:
        trial=directory/('trial_%03d'%spec['ordinal'])
        row,port_rows,service_rows,certificate,support,jobhash,budget=analyze_trial(trial,spec,times,powers)
        if recovery:
            timing=read(trial/'summary.json')
            for began,ended in failure_intervals(recovery['historical_power_errors']):
                if began<timing['measurement_end_monotonic'] and ended>timing['arrival_start_monotonic']-15:
                    raise ValueError('A historical power failure overlaps an included trial window')
        key=(spec['block'],spec['scenario'],spec['workload'])
        if offered.setdefault(key,jobhash)!=jobhash or budgets.setdefault(key,budget)!=budget:
            raise ValueError('Policies did not receive identical jobs and resource constraints')
        records.append(row)
        ports.extend(port_rows)
        services.extend(service_rows)
        certificates.append(certificate)
        supports.append(dict(ordinal=spec['ordinal'],**support))
        for path in sorted(trial.glob('*')):
            if path.is_file():
                hashes[path.relative_to(directory).as_posix()]=sha(path)
    raw=pd.DataFrame(records)
    raw.to_csv(out/'trial_records.csv',index=False)
    pd.DataFrame(ports).to_csv(out/'port_records.csv',index=False)
    service_frame=pd.DataFrame(services)
    service_frame.to_csv(out/'service_records.csv',index=False)
    pd.DataFrame(certificates).to_csv(out/'trace_certificates.csv',index=False)
    (out/'power_support.json').write_text(json.dumps(supports,indent=2),encoding='utf-8')
    groups,contrasts=[],[]
    for (scenario,workload,method),g in raw.groupby(['scenario','workload','method']):
        if set(g.block)!=set(range(10)):
            raise ValueError('Incomplete matched cell')
        row=dict(scenario=scenario,workload=workload,method=method,n=10)
        for metric in PRIMARY+SECONDARY:
            row[metric+'_mean'],row[metric+'_ci95'],row[metric+'_available_blocks']=interval(g[metric])
        groups.append(row)
        if method=='scd':
            continue
        reference=raw[(raw.scenario==scenario)&(raw.workload==workload)&(raw.method=='scd')].set_index('block').sort_index()
        challenger=g.set_index('block').loc[reference.index]
        for metric in PRIMARY+SECONDARY:
            x=(challenger[metric]-reference[metric]).to_numpy()
            mean,half,n=interval(x)
            contrasts.append(dict(scenario=scenario,workload=workload,challenger=method,reference='scd',metric=metric,
                difference_mean=mean,difference_ci95=half,available_pairs=n,p_unadjusted=probability(x),
                scope='primary' if metric in PRIMARY else 'descriptive_secondary',contrast='challenger minus SCD'))
    family=[r for r in contrasts if r['scope']=='primary']
    if len(family)!=40:
        raise ValueError('Primary multiplicity family differs from all ten comparisons and four outcomes')
    holm(family)
    grouped=pd.DataFrame(groups)
    grouped.to_csv(out/'group_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_effects.csv',index=False)
    service_groups=[]
    for (scenario,workload,method,service),g in service_frame.groupby(['scenario','workload','method','service']):
        r=dict(scenario=scenario,workload=workload,method=method,service=service)
        for metric in PRIMARY:
            r[metric+'_mean'],r[metric+'_ci95'],r[metric+'_available_blocks']=interval(g[metric])
        service_groups.append(r)
    pd.DataFrame(service_groups).to_csv(out/'service_summary.csv',index=False)
    tex.mkdir(parents=True,exist_ok=True)
    lines=[r'Condition & Policy & On-time (\%) & Rejection (\%) & Mean (ms) & P95 (ms)\\',r'\midrule']
    power_lines=[r'Condition & Policy & PSE subtotal (J) & Idle-adjusted subtotal (J) & Task charge (J)\\',r'\midrule']
    for scenario,workload,label in CONDITIONS:
        methods=['compute_only','dpp_matched','scd']+(['scd_no_link','scd_independent_network'] if scenario=='network_pressure' else [])
        for method in methods:
            r=grouped[(grouped.scenario==scenario)&(grouped.workload==workload)&(grouped.method==method)].iloc[0]
            cells=[label,LABELS[method]]+[cell(r[m+'_mean'],r[m+'_ci95'],s) for m,s in [('offered_on_time_success',100),('rejection_fraction',100),('mean_latency_ms',1),('p95_latency_ms',1)]]
            lines.append(' & '.join(cells)+r'\\')
            power_lines.append(' & '.join([label,LABELS[method]]+[cell(r[m+'_mean'],r[m+'_ci95']) for m in ['measured_pse_subtotal_j','idle_adjusted_pse_subtotal_j','predicted_incremental_charge_j']])+r'\\')
        lines.append(r'\midrule')
        power_lines.append(r'\midrule')
    lines[-1]=power_lines[-1]=r'\bottomrule'
    (tex/'r2_physical_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (tex/'r2_physical_power_table.tex').write_text('\n'.join(power_lines)+'\n',encoding='utf-8')
    compact=[r'Condition & Compute-first & Matched DPP & SCD\\',r'\midrule']
    for scenario,workload,label in CONDITIONS:
        cells=[label]
        for method in ['compute_only','dpp_matched','scd']:
            r=grouped[(grouped.scenario==scenario)&(grouped.workload==workload)&(grouped.method==method)].iloc[0]
            cells.append(cell(r['offered_on_time_success_mean'],r['offered_on_time_success_ci95'],100))
        compact.append(' & '.join(cells)+r'\\')
    compact.append(r'\bottomrule')
    (tex/'r2_physical_success_table.tex').write_text('\n'.join(compact)+'\n',encoding='utf-8')
    effects=[r'Condition & Comparator & $\Delta$ success (pp) & $p_{\rm Holm}$ & $\Delta$ P95 (ms) & $p_{\rm Holm}$\\',r'\midrule']
    for scenario,workload,label in CONDITIONS:
        methods=['compute_only','dpp_matched']+(['scd_no_link','scd_independent_network'] if scenario=='network_pressure' else [])
        for method in methods:
            cells=[label,LABELS[method]]
            for metric,scale in [('offered_on_time_success',100),('p95_latency_ms',1)]:
                row=next(r for r in contrasts if r['scenario']==scenario and r['workload']==workload
                    and r['challenger']==method and r['metric']==metric)
                # These displayed contrasts follow the prose's SCD-minus-comparator sign.
                cells.extend([cell(-row['difference_mean'],row['difference_ci95'],scale,
                    digits=3 if metric=='offered_on_time_success' else 2),
                    probability_cell(row['p_holm_primary'])])
            effects.append(' & '.join(cells)+r'\\')
        effects.append(r'\midrule')
    effects[-1]=r'\bottomrule'
    (tex/'r2_physical_effects_table.tex').write_text('\n'.join(effects)+'\n',encoding='utf-8')
    mixed=[r'Service & Policy & On-time (\%) & Rejection (\%) & Mean (ms) & P95 (ms)\\',r'\midrule']
    for service in ['mlp','cnn','forest']:
        for method in ['compute_only','dpp_matched','scd']:
            r=next(r for r in service_groups if r['workload']=='mixed' and r['method']==method and r['service']==service)
            cells=[{'mlp':'MLP','cnn':'CNN','forest':'Random forest'}[service],LABELS[method]]
            for metric,scale in [('offered_on_time_success',100),('rejection_fraction',100),('mean_latency_ms',1),('p95_latency_ms',1)]:
                cells.append(cell(r[metric+'_mean'],r[metric+'_ci95'],scale))
            mixed.append(' & '.join(cells)+r'\\')
        mixed.append(r'\midrule')
    mixed[-1]=r'\bottomrule'
    (tex/'r2_physical_services_table.tex').write_text('\n'.join(mixed)+'\n',encoding='utf-8')
    condition_names={('nominal','mlp'):'Nominal',('energy_pressure','mlp'):'Energy',
        ('network_pressure','mlp'):'Network',('nominal','mixed'):'Mixed'}
    method_names={'scd':'SCD','dpp_matched':'DPP','compute_only':'Compute',
        'scd_no_link':'NoPenalty','scd_independent_network':'Independent'}
    metric_names={'offered_on_time_success':('Success',100), 'rejection_fraction':('Rejection',100),
        'mean_latency_ms':('MeanMs',1),'p95_latency_ms':('TailMs',1),
        'measured_pse_subtotal_j':('PSEJ',1),'idle_adjusted_pse_subtotal_j':('IdleAdjustedJ',1),
        'predicted_incremental_charge_j':('ChargeJ',1)}
    entries=[]
    for row in groups:
        condition=condition_names[row['scenario'],row['workload']]
        method=method_names[row['method']]
        for metric,(label,scale) in metric_names.items():
            for ending,field in [('',metric+'_mean'),('CI',metric+'_ci95')]:
                entries.append(('RtwoHw'+condition+method+label+ending,row[field]*scale,2,
                    dict(csv='group_summary.csv',scenario=row['scenario'],workload=row['workload'],method=row['method'],field=field,scale=scale)))
    for row in contrasts:
        if row['metric'] not in metric_names:
            continue
        condition=condition_names[row['scenario'],row['workload']]
        method=method_names[row['challenger']]
        label,scale=metric_names[row['metric']]
        prefix='RtwoHw'+condition+'SCDvs'+method+label
        for ending,field,sign in [('', 'difference_mean',-1),('CI','difference_ci95',1)]:
            entries.append((prefix+ending,row[field]*scale*sign,2,
                dict(csv='paired_effects.csv',scenario=row['scenario'],workload=row['workload'],challenger=row['challenger'],metric=row['metric'],field=field,scale=scale,sign=sign)))
        if row['scope']=='primary':
            entries.append((prefix+'Holm',row['p_holm_primary'],None,
                dict(csv='paired_effects.csv',scenario=row['scenario'],workload=row['workload'],challenger=row['challenger'],metric=row['metric'],field='p_holm_primary')))
    write_macros(tex/'r2_physical_macros.tex',out/'prose_claim_registry.json',entries)
    # Machine-readable claim ledger preserves every source cell used in TeX.
    (out/'table_claim_registry.json').write_text(json.dumps(dict(
        group_source='group_summary.csv',service_source='service_summary.csv',paired_source='paired_effects.csv',
        group_rows=groups,service_rows=service_groups,primary_paired_rows=family,
        paired_table_sign='SCD minus comparator, the opposite of the raw paired CSV convention',
        denominator='All offered individual inferences for success and rejection. Returned parity-valid batches weighted by batch size for latency.',
        confidence='Two-sided Student-t 95% intervals across all ten matched blocks.',
        rounding={'success_percent':2,'paired_success_percentage_points':3,'rejection_percent':2,'latency_ms':2,'energy_j':2}),indent=2),encoding='utf-8')
    for name in ['manifest.json','campaign_status.json','power.jsonl','calibration_parameters.json']:
        hashes[name]=sha(directory/name)
    snapshot=out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    analysis_hashes={}
    for name in ['analyze_r2_physical.py','physical_trace_audit.py','summarize_physical_trials.py','audit_physical_power_sampling.py','r2_evidence_macros.py']:
        path=Path(__file__).with_name(name)
        analysis_hashes[name]=sha(path)
        shutil.copy2(path,snapshot/name)
    report=dict(complete=True,trials=140,randomized_blocks=10,design_cells=len(grouped),
        offered_batches=int(raw.offered_batches.sum()),offered_requests=int(raw.offered_requests.sum()),
        completed_requests=int(raw.completed_requests.sum()),on_time_requests=int(raw.on_time_requests.sum()),
        source_sha256=hashes,analysis_source_sha256=analysis_hashes,parity_failures=0,
        power_samples=len(times),energy_available_trials=sum(s['energy_available'] for s in supports),
        metered_ports=manifest['metered_ports'],power_boundary='Fixed seven-port PSE subtotal, not 14-board total. 75 s active and separate final 15 s idle baseline.',
        primary_tests=40,multiplicity='Global Holm across four primary outcomes and all ten challenger-minus-SCD contrasts.',
        latency_denominator='Fully returned parity-valid batches, weighted by individual predictions. All rejected/failed/censored requests remain offered-success failures.',
        missing_metrics='Undefined or incompletely observed groups retain all ten rows, with no pooled CI and conservative p=1. No selected complete-case estimate.',
        allocation='Application-identity-coherent tenant groups, fixed before collecting formal v4.',
        commissioning='All pre-v4 R2 trials excluded as a whole after input-routing and tenant-definition corrections. They are retained separately.')
    if recovery:
        report['acquisition_recovery']=dict(retained_blocks=recovery['retained_complete_blocks'],
            repeated_from_block=recovery['first_repeated_block'],
            excluded_parent_completed_ordinals=recovery['excluded_parent_completed_ordinals'],
            interrupted_parent_block_retained_separately=True,
            same_execution_sources_and_inputs=True,included_windows_overlap_no_historical_power_failure=True,
            historical_timeouts=len(recovery['historical_power_errors']),
            instrumentation_amendment=recovery.get('instrumentation_amendment'),
            resumption_sha256=sha(directory/'resumption.json'))
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','analysis_source_sha256']},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--tex',type=Path,default=Path(__file__).resolve().parents[2]/'manuscript/sections')
    a=p.parse_args()
    summarize(a.directory,a.tex,a.out)
