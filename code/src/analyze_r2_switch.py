"""Direct shared/disjoint Ethernet contention evidence from all 60 trials."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
from src.analyze_r2_physical import interval, probability, holm, cell, probability_cell, origin_verifier

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_counters(text):
    result={}
    for direction in ['Input','Output']:
        match=re.search(direction+r'\s*\(total\):\s*(\d+) packets,\s*(\d+) bytes',text,re.I)
        if not match:
            raise ValueError('Interface total counters unavailable for '+direction)
        result[direction.lower()+'_packets']=int(match[1])
        result[direction.lower()+'_bytes']=int(match[2])
    speed=re.search(r'(\d+)Mbps-speed mode',text)
    if not speed:
        raise ValueError('Negotiated interface rate unavailable')
    result['line_rate_mbps']=int(speed[1])
    # Keep full replies as evidence. Missing optional firmware counters are NA.
    for name,pattern in [('input_errors',r'(\d+) input errors'),('output_errors',r'(\d+) output errors'),
                         ('drops',r'(\d+) drops')]:
        match=re.search(pattern,text,re.I)
        result[name]=int(match[1]) if match else None
    return result

def summarize(directory,tex,out=None):
    protocol=read(directory/'protocol.json')
    completion=read(directory/'completion.json')
    if not completion['complete'] or completion['trials']!=60 or len(protocol['trials'])!=60:
        raise ValueError('The full 60-trial paired design is required')
    for name,digest in protocol['source_sha256'].items():
        if sha(directory/'source_snapshot'/name)!=digest:
            raise ValueError('Execution source snapshot changed')
    recovery=None
    if (directory/'resumption.json').is_file():
        recovery=read(directory/'resumption.json')
        verify_origin=origin_verifier(directory)
        retained=6*recovery['retained_complete_blocks']
        expected={'trial_%03d.json'%ordinal for ordinal in range(retained)}
        if set(recovery['retained_trial_sha256'])!=expected or recovery['first_repeated_block']!=retained//6:
            raise ValueError('Recovery did not retain complete six-condition blocks')
        verify_origin(directory/'protocol.json',recovery['parent_protocol_sha256'])
        for name,digest in recovery['retained_trial_sha256'].items():
            verify_origin(directory/name,digest)
        for name,digest in recovery['additional_source_sha256'].items():
            verify_origin(directory/'resumption_sources'/name,digest)
    out=out or directory/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    records,counters,hashes=[],[],{}
    for spec in protocol['trials']:
        path=directory/('trial_%03d.json'%spec['ordinal'])
        trial=read(path)
        if not trial['complete'] or trial['spec']!=spec:
            raise ValueError('Incomplete or mismatched trial')
        hashes[path.name]=sha(path)
        probe=trial['probe']['result']
        samples=probe['probes']
        if len(samples)!=probe['sent'] or len({s['sequence'] for s in samples})!=len(samples):
            raise ValueError('Probe accounting failed')
        good=[s['rtt_s'] for s in samples if s['rtt_s'] is not None]
        if len(good)!=probe['received'] or any(x<0 for x in good):
            raise ValueError('RTT result mismatch')
        load=trial['load']['result']
        elapsed=load['elapsed_s']
        if not (33<=elapsed<=40) or probe['duration_s']!=30:
            raise ValueError('Bounded experiment duration changed')
        record=dict(ordinal=spec['ordinal'],block=spec['block'],path=spec['path'],target_wire_mbps=spec['wire_mbps'],
            sent=probe['sent'],received=probe['received'],loss_fraction=1-probe['received']/probe['sent'],
            mean_rtt_ms=float(np.mean(good)*1000) if good else None,
            p50_rtt_ms=float(np.quantile(good,.5,method='inverted_cdf')*1000) if good else None,
            p95_rtt_ms=float(np.quantile(good,.95,method='inverted_cdf')*1000) if good else None,
            p99_rtt_ms=float(np.quantile(good,.99,method='inverted_cdf')*1000) if good else None,
            actual_wire_mbps=8*load['wire_bytes']/elapsed/1e6,actual_payload_mbps=8*load['payload_bytes']/elapsed/1e6,
            server7_packets=trial['servers']['7']['result']['packets'],server23_packets=trial['servers']['23']['result']['packets'])
        if not record['sent']:
            raise ValueError('No probes sent')
        for port in [1,3,7,23]:
            key='display interface GigabitEthernet1/0/'+str(port)
            before=trial['before'][key]
            after=trial['after'][key]
            first=parse_counters(before['reply'])
            last=parse_counters(after['reply'])
            if first['line_rate_mbps']!=last['line_rate_mbps'] or (port==7 and first['line_rate_mbps']!=100):
                raise ValueError('Physical bottleneck link rate changed')
            duration=(after['query_start']+after['query_end']-before['query_start']-before['query_end'])/2
            row=dict(ordinal=spec['ordinal'],block=spec['block'],path=spec['path'],target_wire_mbps=spec['wire_mbps'],
                     port=port,interval_s=duration,line_rate_mbps=first['line_rate_mbps'])
            for metric in ['input_packets','input_bytes','output_packets','output_bytes','input_errors','output_errors','drops']:
                delta=last[metric]-first[metric] if first[metric] is not None and last[metric] is not None else None
                if delta is not None and delta<0:
                    raise ValueError('Interface counter reset or wrap')
                row['delta_'+metric]=delta
            row['observed_output_mbps']=row['delta_output_bytes']*8/duration/1e6
            row['observed_input_mbps']=row['delta_input_bytes']*8/duration/1e6
            counters.append(row)
        records.append(record)
    raw=pd.DataFrame(records)
    raw.to_csv(out/'trial_records.csv',index=False)
    pd.DataFrame(counters).to_csv(out/'interface_counter_deltas.csv',index=False)
    groups=[]
    metrics=['actual_wire_mbps','loss_fraction','mean_rtt_ms','p50_rtt_ms','p95_rtt_ms','p99_rtt_ms']
    for (path,load),g in raw.groupby(['path','target_wire_mbps']):
        if set(g.block)!=set(range(10)):
            raise ValueError('Incomplete switch matched blocks')
        r=dict(path=path,target_wire_mbps=load)
        for metric in metrics:
            r[metric+'_mean'],r[metric+'_ci95'],r[metric+'_available_blocks']=interval(g[metric])
        groups.append(r)
    grouped=pd.DataFrame(groups)
    grouped.to_csv(out/'group_summary.csv',index=False)
    contrasts=[]
    for load in [85.,105.]:
        shared=raw[(raw.path=='shared')&(raw.target_wire_mbps==load)].set_index('block').sort_index()
        disjoint=raw[(raw.path=='disjoint')&(raw.target_wire_mbps==load)].set_index('block').loc[shared.index]
        for metric in ['p95_rtt_ms','loss_fraction']:
            x=(shared[metric]-disjoint[metric]).to_numpy()
            mean,half,n=interval(x)
            contrasts.append(dict(target_wire_mbps=load,metric=metric,contrast='shared minus disjoint',
                difference_mean=mean,difference_ci95=half,available_pairs=n,p_unadjusted=probability(x)))
    holm(contrasts)
    pd.DataFrame(contrasts).to_csv(out/'paired_effects.csv',index=False)
    lines=[r'Load path & Target (Mb/s) & Sent wire (Mb/s) & RTT P50 (ms) & RTT P95 (ms) & Loss (\%)\\',r'\midrule']
    for path,load in [('shared',0),('shared',50),('shared',85),('shared',105),('disjoint',85),('disjoint',105)]:
        r=grouped[(grouped.path==path)&(grouped.target_wire_mbps==load)].iloc[0]
        cells=[path.capitalize(),str(load)]
        for metric,scale in [('actual_wire_mbps',1),('p50_rtt_ms',1),('p95_rtt_ms',1),('loss_fraction',100)]:
            digits=5 if metric=='actual_wire_mbps' else 3 if metric.startswith('p') else 2
            cells.append(cell(r[metric+'_mean'],r[metric+'_ci95'],scale,digits=digits))
        lines.append(' & '.join(cells)+r'\\')
    lines.append(r'\bottomrule')
    tex.mkdir(parents=True,exist_ok=True)
    (tex/'r2_switch_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    effects=[r'Target load (Mb/s) & Outcome & Shared minus disjoint & $p_{\rm Holm}$\\',r'\midrule']
    for row in contrasts:
        label,scale=('P95 RTT (ms)',1) if row['metric']=='p95_rtt_ms' else ('Probe loss (pp)',100)
        effects.append(' & '.join([str(int(row['target_wire_mbps'])),label,
            cell(row['difference_mean'],row['difference_ci95'],scale,
                digits=3 if row['metric']=='p95_rtt_ms' else 2),
            probability_cell(row['p_holm_primary'])])+r'\\')
    effects.append(r'\bottomrule')
    (tex/'r2_switch_effects_table.tex').write_text('\n'.join(effects)+'\n',encoding='utf-8')
    from src.r2_evidence_macros import write_macros
    load_names={0:'Zero',50:'Fifty',85:'EightyFive',105:'HundredFive'}
    names={'actual_wire_mbps':('Rate',1),'loss_fraction':('Loss',100),
        'mean_rtt_ms':('MeanMs',1),'p50_rtt_ms':('MedianMs',1),'p95_rtt_ms':('TailMs',1)}
    entries=[]
    for row in groups:
        prefix='RtwoSwitch'+row['path'].capitalize()+load_names[row['target_wire_mbps']]
        for metric,(label,scale) in names.items():
            for ending,field in [('',metric+'_mean'),('CI',metric+'_ci95')]:
                digits=5 if label=='Rate' else 3 if label.endswith('Ms') else 2
                entries.append((prefix+label+ending,row[field]*scale,digits,
                    dict(csv='group_summary.csv',path=row['path'],target_wire_mbps=row['target_wire_mbps'],field=field,scale=scale)))
    for row in contrasts:
        label,scale=names[row['metric']]
        prefix='RtwoSwitchSharedDifference'+load_names[row['target_wire_mbps']]+label
        for ending,field,multiplier,digits in [('', 'difference_mean',scale,2),('CI','difference_ci95',scale,2),('Holm','p_holm_primary',1,None)]:
            if label.endswith('Ms') and ending!='Holm':
                digits=3
            entries.append((prefix+ending,row[field]*multiplier,digits,
                dict(csv='paired_effects.csv',target_wire_mbps=row['target_wire_mbps'],metric=row['metric'],field=field,scale=multiplier)))
    entries.append(('RtwoSwitchSentProbes',int(raw.sent.sum()),0,dict(csv='trial_records.csv',operation='sum sent')))
    write_macros(tex/'r2_switch_macros.tex',out/'prose_claim_registry.json',entries)
    audit=dict(complete=True,trials=60,randomized_blocks=10,probes_sent=int(raw.sent.sum()),probes_received=int(raw.received.sum()),
        source_sha256=hashes,protocol_sha256=sha(directory/'protocol.json'),
        primary_contrasts=4,multiplicity='Holm across both shared-minus-disjoint contrasts and both primary outcomes',
        boundary='End-to-end board-to-board RTT and loss on the measured switch path. Host/NIC queues contribute. No internal buffer-residence observation is inferred.',
        rates='Actual sender wire rate includes 66 overhead bytes per 1400-byte UDP payload. Switch counter interval also includes setup and controller traffic.',
        clocks='Controller monotonic counter-query times and same-host probe RTT. Switch calendar timestamps are not used.',
        quantiles='Left-inverse empirical CDF of returned probes in each trial. Probe loss reported separately.')
    if recovery:
        audit['acquisition_recovery']=dict(retained_complete_blocks=recovery['retained_complete_blocks'],
            first_repeated_block=recovery['first_repeated_block'],
            excluded_parent_complete_ordinals=recovery['excluded_parent_complete_ordinals'],
            resumption_sha256=sha(directory/'resumption.json'),
            connection_records_sha256=sha(directory/'management_connections.jsonl'),
            amendment=recovery['amendment'])
    snap=out/'source_snapshot'
    snap.mkdir(exist_ok=True)
    audit['analysis_source_sha256']={}
    for name in ['analyze_r2_switch.py','analyze_r2_physical.py','r2_evidence_macros.py']:
        source=Path(__file__).with_name(name)
        shutil.copy2(source,snap/name)
        audit['analysis_source_sha256'][name]=sha(source)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(grouped.to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--tex',type=Path,default=Path(__file__).resolve().parents[2]/'manuscript/sections')
    a=p.parse_args()
    summarize(a.directory,a.tex,a.out)
