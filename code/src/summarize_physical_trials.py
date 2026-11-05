"""Complete-run trial statistics, measured power and actual-state certificates.

Refuses a partial design. This creates evidence tables, not a submission-ready
claim; container-specific evidence and manuscript interpretation remain separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.stats import t
from src.physical_trace_audit import audit_trace


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def weighted_quantile(values, weights, fraction):
    values,weights=np.asarray(values,dtype=float),np.asarray(weights,dtype=float)
    if len(values)==0 or len(values)!=len(weights) or np.any(weights<=0):
        raise ValueError('Invalid weighted empirical distribution')
    order=np.argsort(values,kind='stable')
    index=np.searchsorted(np.cumsum(weights[order]),fraction*weights.sum(),side='left')
    return float(values[order[min(index,len(values)-1)]])


def integrate_observed(times, values, start, stop):
    times,values=np.asarray(times,dtype=float),np.asarray(values,dtype=float)
    if (stop<=start or len(times)<2 or len(times)!=len(values)
            or np.any(np.diff(times)<=0) or not np.isfinite(times).all()):
        raise ValueError('Invalid power interval or sampling clock')
    # Interpolate the leading boundary only when it is bracketed by a normal
    # sampling interval.  This window is never allowed to span a reconnect.
    left_start=int(np.searchsorted(times,start,side='right'))-1
    right_start=int(np.searchsorted(times,start,side='left'))
    if left_start<0 or right_start>=len(times):
        raise ValueError('Power interval lacks observed support at its start')
    if times[right_start]!=start and times[right_start]-times[left_start]>2.5:
        raise ValueError('Power interval starts inside a sampling gap above 2.5 s')
    inside_indices=np.flatnonzero((times>start)&(times<stop))
    last_before_stop=int(np.searchsorted(times,stop,side='right'))-1
    if last_before_stop<0 or stop-times[last_before_stop]>2.5:
        raise ValueError('Power interval lacks timely observed support at its stop')
    x=np.concatenate([[start],times[inside_indices],[stop]])
    if np.max(np.diff(x))>2.5:
        raise ValueError('Power interval has a sampling gap above 2.5 s')
    start_value=float(np.interp(start,times,values))
    right_stop=int(np.searchsorted(times,stop,side='left'))
    # A planned post-trial authentication pause can make the next sample more
    # than 2.5 s away even though the measured interval itself is continuous.
    # In that case (and at the campaign's final stop), carry the most recent
    # observation only over the sub-sample terminal tail.  Otherwise retain
    # ordinary linear interpolation at the boundary.
    if right_stop<len(times) and times[right_stop]==stop:
        stop_value=float(values[right_stop])
    elif (right_stop<len(times)
          and times[right_stop]-times[last_before_stop]<=2.5):
        stop_value=float(np.interp(stop,times,values))
    else:
        stop_value=float(values[last_before_stop])
    y=np.concatenate([[start_value],values[inside_indices],[stop_value]])
    if not np.isfinite(y).all():
        raise ValueError('Power reading missing inside measurement window')
    return float(np.trapezoid(y,x))


def interval(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)==0:
        return None,None,0
    if len(values)!=10:
        raise ValueError('Expected all ten matched blocks, not a partial/pseudoreplicated CI')
    return float(values.mean()),float(t.ppf(.975,9)*values.std(ddof=1)/np.sqrt(10)),10


def summarize(directory):
    manifest=read_json(directory/'manifest.json')
    campaign=read_json(directory/'campaign_status.json')
    if not campaign['complete'] or len(campaign['completed'])!=280 or manifest['planned_trials']!=280:
        raise ValueError('Formal physical matrix is not complete')
    if campaign['power_sampler']['errors'] or not campaign['power_sampler']['thread_stopped']:
        raise ValueError('Power acquisition was incomplete')
    out=directory/'analysis'
    out.mkdir(exist_ok=True)
    power_times=[]
    power_values=[]
    with (directory/'power.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            sample=json.loads(line)
            values={p['port']:p['poe_output_w'] for p in sample['ports']}
            power_times.append((sample['query_start']+sample['query_end'])/2.)
            power_values.append([values.get(p) for p in [5,6,7,8,9]])
    power_times=np.asarray(power_times)
    power_values=np.asarray(power_values,dtype=float)
    if np.any(np.diff(power_times)<=0):
        raise ValueError('Unordered power clock')
    trial_rows,port_rows,certificates,hashes=[],[],[],{}
    offered_hashes={}
    for spec in manifest['trials']:
        trial=directory/('trial_%03d'%spec['ordinal'])
        summary=read_json(trial/'summary.json')
        if not summary['complete'] or summary['error'] or summary['parity_failures']:
            raise ValueError('Incomplete or numerically invalid trial %s'%spec['ordinal'])
        if any(c['errors'] or not c['stop_acknowledged'] or not c['directory_removed']
               or not c.get('pipeline_threads_stopped',True) for c in summary['cleanup'].values()):
            raise ValueError('Worker cleanup not confirmed')
        protocol=read_json(trial/'protocol.json')
        key=(spec['block'],spec['cohort'],spec['scenario'])
        previous=offered_hashes.setdefault(key,protocol['offered_sha256'])
        if previous!=protocol['offered_sha256']:
            raise ValueError('Policies did not receive matched offered work')
        jobs=read_json(trial/'offered_jobs.json')
        events=[json.loads(s) for s in (trial/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        parameters=read_json(trial/'model_parameters.json')
        audit=audit_trace(events,parameters,spec)
        certificates.append(dict(ordinal=spec['ordinal'],block=spec['block'],cohort=spec['cohort'],
                                 scenario=spec['scenario'],method=spec['method'],**audit))
        dispatched={}
        results={}
        rejected=set()
        decision_times=[]
        dispatch_times=[]
        port_counts=Counter()
        temperature={p:[] for p in summary['ports']}
        for event in events:
            kind=event['event']
            if kind=='decision':
                identity=event['job']['id']
                decision_times.append(event['decision_s'])
                dispatch_times.append(event['dispatch_path_s'])
                selected=event['decision']['selected']
                if selected is None or 'submission_error' in event:
                    rejected.add(identity)
                else:
                    candidate=event['decision']['candidates'][selected]
                    dispatched[identity]=candidate
                    port_counts[candidate['port']]+=event['job']['batch']
            elif kind=='rejected':
                rejected.add(event['job']['id'])
            elif kind=='result':
                identity=event['job']['id']
                if identity in results:
                    raise ValueError('Duplicate result identity')
                result=event['result']
                results[identity]=result
                for endpoint in ['before','after']:
                    value=result.get(endpoint,{}).get('cpu_c')
                    if value is not None and np.isfinite(value):
                        temperature[result['port']].append(value)
            elif kind=='state':
                for port,value in zip(summary['ports'],event['temperature_c']):
                    if value is not None and np.isfinite(value):
                        temperature[port].append(value)
        offered={j['id'] for j in jobs}
        if (set(dispatched)&rejected or set(dispatched)|rejected!=offered
                or not set(results)<=set(dispatched)):
            raise ValueError('Dropped/duplicated offered-work accounting')
        censored=set(summary['outstanding_at_fixed_drain'])
        if set(dispatched)-set(results)!=censored:
            raise ValueError('Unreported drain censoring')
        latencies=[]
        weights=[]
        on_time=0
        successes=0
        total_requests=sum(j['batch'] for j in jobs)
        for job in jobs:
            result=results.get(job['id'])
            if result and result['batch_atomic_success']:
                if not result['parity_ok'] or len(result['predictions'])!=job['batch']:
                    raise ValueError('Invalid successful result')
                latencies.append(result['release_to_result_s'])
                weights.append(job['batch'])
                successes+=job['batch']
                on_time+=job['batch']*int(result['release_to_result_s']<=job['deadline_s'])
        if successes!=summary['successful_inferences']:
            raise ValueError('Completion count does not match raw results')
        row=dict(ordinal=spec['ordinal'],block=spec['block'],cohort=spec['cohort'],scenario=spec['scenario'],
            method=spec['method'],offered_batches=len(jobs),offered_requests=total_requests,
            admitted_requests=sum(j['batch'] for j in jobs if j['id'] in dispatched),
            completed_requests=successes,on_time_requests=on_time,offered_on_time_success=on_time/total_requests,
            rejection_fraction=sum(j['batch'] for j in jobs if j['id'] in rejected)/total_requests,
            conditional_miss_fraction=(successes-on_time)/successes if successes else None,
            mean_latency_ms=float(np.average(latencies,weights=weights)*1000) if successes else None,
            p95_latency_ms=weighted_quantile(latencies,weights,.95)*1000 if successes else None,
            mean_decision_ms=float(np.mean(decision_times)*1000),
            p95_dispatch_path_ms=float(np.quantile(dispatch_times,.95)*1000),
            guard_batches=summary['guarded_batches'],censored_batches=len(censored),failed_batches=summary['failed_batches'],
            predicted_incremental_charge_j=sum(c['predicted_energy_j'] for c in dispatched.values()),
            peak_observed_cpu_c=max((max(v) for v in temperature.values() if v),default=None),
            measured_pse_energy_j=None,baseline_adjusted_pse_energy_j=None,idle_pse_w=None,
            max_queue_state_reconstruction_error_j=audit['max_state_error_j'])
        if spec['cohort']=='poe':
            begin,end=summary['arrival_start_monotonic'],summary['measurement_end_monotonic']
            if abs(end-begin-75.)>1e-5:
                raise ValueError('Unequal physical energy windows')
            total_power=power_values.sum(axis=1)
            row['measured_pse_energy_j']=integrate_observed(power_times,total_power,begin,end)
            row['idle_pse_w']=integrate_observed(power_times,total_power,begin-15.,begin)/15.
            # Do not clip a negative baseline-adjusted residual: changes in
            # cooling/rounding can make it negative without negative supplied power.
            row['baseline_adjusted_pse_energy_j']=row['measured_pse_energy_j']-75.*row['idle_pse_w']
        for port in summary['ports']:
            port_rows.append(dict(ordinal=spec['ordinal'],block=spec['block'],cohort=spec['cohort'],
                scenario=spec['scenario'],method=spec['method'],port=port,
                assigned_requests=port_counts[port],share_of_admitted=port_counts[port]/max(row['admitted_requests'],1),
                peak_observed_cpu_c=max(temperature[port]) if temperature[port] else None,
                guard_batches=sum(r['port']==port and r['status']=='temperature_guard' for r in results.values())))
        trial_rows.append(row)
        for filename in ['events.jsonl','summary.json','protocol.json','model_parameters.json']:
            hashes[str(spec['ordinal'])+'/'+filename]=hashlib.sha256((trial/filename).read_bytes()).hexdigest()
    raw=pd.DataFrame(trial_rows)
    raw.to_csv(out/'trial_records.csv',index=False)
    pd.DataFrame(port_rows).to_csv(out/'port_records.csv',index=False)
    pd.DataFrame(certificates).to_csv(out/'trace_certificates.csv',index=False)
    metrics=['offered_on_time_success','rejection_fraction','mean_latency_ms','p95_latency_ms',
             'measured_pse_energy_j','baseline_adjusted_pse_energy_j','idle_pse_w',
             'predicted_incremental_charge_j','peak_observed_cpu_c','mean_decision_ms',
             'p95_dispatch_path_ms','guard_batches','censored_batches']
    grouped=[]
    paired=[]
    for (cohort,scenario,method),group in raw.groupby(['cohort','scenario','method']):
        if len(group)!=10 or group.block.nunique()!=10:
            raise ValueError('Incomplete matched design cell')
        record=dict(cohort=cohort,scenario=scenario,method=method,n=10)
        reference=raw[(raw.cohort==cohort)&(raw.scenario==scenario)&(raw.method=='scd')].set_index('block')
        matched=group.set_index('block').sort_index()
        reference=reference.loc[matched.index]
        comparison=dict(cohort=cohort,scenario=scenario,method=method,contrast='method minus SCD',n=10)
        for metric in metrics:
            mean,half,n=interval(group[metric])
            record[metric+'_mean'],record[metric+'_ci95'],record[metric+'_n']=mean,half,n
            mean,half,n=interval(matched[metric]-reference[metric])
            comparison[metric+'_mean'],comparison[metric+'_ci95']=mean,half
        grouped.append(record)
        paired.append(comparison)
    grouped=pd.DataFrame(grouped)
    grouped.to_csv(out/'group_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_effects.csv',index=False)
    labels={'compute_only':'Compute-first','energy_greedy':'Energy-first','dpp_matched':'Matched DPP',
            'scd':'SCD','scd_no_link':'No link','scd_no_debt':'No debt','scd_no_guard':'No guard','scd_no_thermal':'No thermal'}
    for cohort in ['poe','fleet']:
        lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',
               r'Scenario & Method & On-time (\%) & Mean (ms) & P95 (ms) & PSE energy (J)\\',r'\midrule']
        for r in grouped[grouped.cohort==cohort].to_dict('records'):
            cells=[r['scenario'].replace('_',r'\_'),labels[r['method']]]
            for metric,scale in [('offered_on_time_success',100),('mean_latency_ms',1),('p95_latency_ms',1),('measured_pse_energy_j',1)]:
                value=r[metric+'_mean']
                cells.append(r'\textemdash' if value is None or not np.isfinite(value) else
                             '$%.2f\\pm%.2f$'%(value*scale,r[metric+'_ci95']*scale))
            lines.append(' & '.join(cells)+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (out/('physical_'+cohort+'_table.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    power_hash=hashlib.sha256((directory/'power.jsonl').read_bytes()).hexdigest()
    sampling_path=out/'power_sampling_audit.json'
    if not sampling_path.exists():
        raise ValueError('Power sampling audit is required before final statistics')
    sampling=read_json(sampling_path)
    if (sampling.get('power_jsonl_sha256')!=power_hash
            or sampling.get('power_sample_count')!=len(power_times)
            or sampling.get('poe_trial_count')!=160
            or sampling.get('poe_trials_affected')!=0):
        raise ValueError('Power sampling audit does not certify this complete campaign')
    analysis_sources=['summarize_physical_trials.py','physical_trace_audit.py',
                      'audit_physical_power_sampling.py']
    analysis_hashes={}
    source_snapshot=out/'source_snapshot'
    source_snapshot.mkdir(exist_ok=True)
    for name in analysis_sources:
        source=Path(__file__).with_name(name)
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        target=source_snapshot/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=source_hash:
            raise ValueError('Existing analysis source snapshot differs: '+name)
        if not target.exists():
            shutil.copy2(source,target)
        analysis_hashes[name]=source_hash
    audit=dict(complete=True,trials=len(raw),groups=len(grouped),blocks=10,
        successful_individual_inferences=int(raw.completed_requests.sum()),
        offered_individual_inferences=int(raw.offered_requests.sum()),
        parity_failures=0,power_samples=len(power_times),
        energy_boundary='Five calibrated PSE output ports; fixed 75 s; no externally powered whole-fleet energy claim',
        power_integration=('Timestamped trapezoidal quadrature; a terminal last-observation hold is used only '
                           'when the post-window sample is separated by a planned authentication pause, with '
                           'the in-window terminal tail required to be no longer than 2.5 s'),
        quantile='Request-weighted left inverse empirical CDF across fully returned parity-valid batches',
        intervals='Two-sided Student-t across ten randomized matched blocks; not across individual requests',
        failures='Rejected, failed, guarded and fixed-drain censored requests remain offered-service failures',
        certificate='Actual live state/credit reconstruction and same-state shadow comparison; not own-trajectory optimality',
        power_sampling_audit_sha256=hashlib.sha256(sampling_path.read_bytes()).hexdigest(),
        power_sampling_audit=dict(
            global_gap_count=sampling['global_gap_count'],
            global_gaps_crossing_session_boundary=sampling['global_gaps_crossing_session_boundary'],
            poe_trials_affected=sampling['poe_trials_affected'],
            terminal_hold_count=sampling['terminal_hold_count'],
            maximum_terminal_hold_s=sampling['maximum_terminal_hold_s']),
        trial_file_sha256=hashes,
        manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
        power_sha256=power_hash,
        analysis_source_sha256=analysis_hashes['summarize_physical_trials.py'],
        analysis_source_sha256_all=analysis_hashes)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='trial_file_sha256'}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    while args.wait and not (args.directory/'campaign_status.json').exists():
        time.sleep(30.)
    summarize(args.directory)


if __name__=='__main__':
    main()
