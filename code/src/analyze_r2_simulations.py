"""Complete factorial statistics, global Holm correction, and source-linked TeX."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/r2/simulation_factorial_20261105'
OUT=RUN/'analysis'
TEX=ROOT.parent/'manuscript/sections'
BASELINE=ROOT/'results/strict/campaign_formula_v7_20260914/campaign_results.csv'
METRICS=['success_ratio','task_energy_mj','supplied_energy_mj','p95_latency_ms','mean_peak_aoi_ms','jain_tenant_success','minimum_tenant_success_ratio']
LABELS={'scd':'SCD','dpp_matched':'Matched DPP','compute_first':'Compute-first','scd_no_debt':'No explicit debt prices',
        'scd_no_debt_feedback':'No debt feedback','scd_no_forecast':'No forecast','scd_static_weights':'Static weights',
        'scd_elastic_energy':'Elastic allocation','scd_shared_energy':'Fully shared'}
def interval(x):
    x=np.asarray(x,float)
    if len(x)!=5:
        raise AssertionError('Expected all five paired traces')
    if not np.isfinite(x).all():
        if np.isnan(x).all():
            return float('nan'),float('nan')
        raise AssertionError('Partly missing metric')
    return float(x.mean()),float(t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5))
def fmt(mean,half,digits=2,scale=1):
    if not math.isfinite(mean):
        return r'\textemdash'
    return '$'+f'{mean*scale:.{digits}f}'+r'\pm'+f'{half*scale:.{digits}f}'+'$'
def validate_sources(data,protocol):
    expected={task['key']:task for task in protocol['tasks']}
    if data.key.duplicated().any() or len(expected)!=200:
        raise AssertionError('Repeated result or registered key')
    hashes={}
    for name,value in protocol['execution_source_sha256'].items():
        path=RUN/'source_snapshot'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:
            raise AssertionError('Execution snapshot differs from protocol: '+name)
    frame=data.set_index('key')
    for path in sorted((RUN/'runs').glob('*.json')):
        raw=json.loads(path.read_text(encoding='utf-8'))
        if raw['key'] not in expected or raw['key'] in hashes:
            raise AssertionError('Unregistered or repeated raw run')
        task=expected[raw['key']]
        if raw['method']!=task['method'] or raw['scenario_parameters']!=task['scenario']:
            raise AssertionError('Executed scenario differs from registry')
        row=frame.loc[raw['key']]
        for field,value in raw.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if not np.isclose(float(row[field]),float(value),rtol=2e-13,atol=2e-12,equal_nan=True):
                    raise AssertionError('CSV differs from raw record: '+raw['key']+' '+field)
        if raw['admitted_requests']+raw['rejected_requests']!=raw['evaluated_requests']:
            raise AssertionError('Request conservation failed')
        if sum(raw['tenant_success'])!=raw['on_time_requests'] or sum(raw['tenant_offered'])!=raw['evaluated_requests']:
            raise AssertionError('Tenant counts do not conserve outcomes')
        if not np.isclose(raw['success_ratio'],raw['on_time_requests']/raw['evaluated_requests'],atol=1e-15,rtol=0):
            raise AssertionError('Offered-success denominator differs')
        for field,value in [('deployment_sha256',protocol['deployment_sha256']),
                            ('simulator_sha256',protocol['execution_source_sha256']['strict_simulate.py']),
                            ('scheduler_core_sha256',protocol['execution_source_sha256']['scheduler_core.py']),
                            ('service_forecast_sha256',protocol['forecast_sha256'])]:
            if raw[field]!=value:
                raise AssertionError('Mixed execution source: '+field)
        hashes[raw['key']]=hashlib.sha256(path.read_bytes()).hexdigest()
    if set(hashes)!=set(expected):
        raise AssertionError('Missing raw result records')
    return hashes
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    TEX.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((RUN/'protocol.json').read_text())
    completion=json.loads((RUN/'completion.json').read_text())
    data=pd.read_csv(RUN/'campaign_results.csv')
    if not completion['complete'] or len(data)!=200 or set(data.key)!=set(x['key'] for x in protocol['tasks']):
        raise AssertionError('Incomplete or unregistered runs')
    raw_hashes=validate_sources(data,protocol)
    old=pd.read_csv(BASELINE)
    regressions=[]
    for method in ['scd','dpp_matched','compute_first']:
        a=data[(data.scenario=='r2_nominal')&(data.method==method)].set_index('seed').sort_index()
        b=old[(old.scenario=='primary')&(old.method==method)].set_index('seed').loc[a.index]
        for metric in METRICS+['on_time_requests','evaluated_requests','rejected_requests']:
            if not np.allclose(a[metric],b[metric],atol=1e-10,rtol=1e-12,equal_nan=True):
                raise AssertionError('R1 baseline behavior changed: '+method+' '+metric)
        regressions.append(dict(method=method,seeds=5,all_metrics_identical=True))
    groups=[]
    for (scenario,method),g in data.groupby(['scenario','method']):
        row=dict(scenario=scenario,method=method,n=5)
        for metric in METRICS:
            row[metric+'_mean'],row[metric+'_ci95']=interval(g[metric])
        row['no_service']=bool((g.on_time_requests==0).all())
        groups.append(row)
    summary=pd.DataFrame(groups)
    summary.to_csv(OUT/'group_summary.csv',index=False)
    contrasts=[]
    for scenario in ['r2_nominal','r2_energy_03','r2_energy_01','r2_network_100']:
        base=data[(data.scenario==scenario)&(data.method=='scd')].set_index('seed').sort_index()
        for challenger in [m for m in LABELS if m!='scd']+['no_energy_guard']:
            other=data[(data.scenario==(scenario+'_no_energy_guard' if challenger=='no_energy_guard' else scenario))&
                       (data.method==('scd' if challenger=='no_energy_guard' else challenger))].set_index('seed').loc[base.index]
            for metric in METRICS:
                x=(other[metric]-base[metric]).to_numpy()
                mean,half=interval(x)
                pvalue=1. if not np.isfinite(x).all() or np.all(x==0) else 0. if np.std(x)==0 else float(ttest_1samp(x,0.).pvalue)
                contrasts.append(dict(scenario=scenario,challenger=challenger,reference='scd',metric=metric,n=5,
                    difference_mean=mean,difference_ci95=half,p_unadjusted=pvalue,comparison='challenger minus reference'))
        elastic=data[(data.scenario==scenario)&(data.method=='scd_elastic_energy')].set_index('seed').loc[base.index]
        shared=data[(data.scenario==scenario)&(data.method=='scd_shared_energy')].set_index('seed').loc[base.index]
        for metric in METRICS:
            x=(elastic[metric]-shared[metric]).to_numpy()
            mean,half=interval(x)
            pvalue=1. if not np.isfinite(x).all() or np.all(x==0) else 0. if np.std(x)==0 else float(ttest_1samp(x,0.).pvalue)
            contrasts.append(dict(scenario=scenario,challenger='scd_elastic_energy',reference='scd_shared_energy',metric=metric,n=5,
                difference_mean=mean,difference_ci95=half,p_unadjusted=pvalue,comparison='challenger minus reference'))
    ordered=sorted(range(len(contrasts)),key=lambda i:contrasts[i]['p_unadjusted'])
    adjusted=0.
    for rank,index in enumerate(ordered):
        adjusted=max(adjusted,min(1.,(len(contrasts)-rank)*contrasts[index]['p_unadjusted']))
        contrasts[index]['p_holm_global']=adjusted
    paired=pd.DataFrame(contrasts)
    paired.to_csv(OUT/'paired_effects.csv',index=False)
    lines=[r'Condition & Policy & Success (\%) & Task (MJ) & P95 (s) & Jain & Minimum tenant (\%) \\',r'\midrule']
    condlabels={'r2_nominal':'Nominal','r2_energy_03':'Budget 0.30','r2_energy_01':'Budget 0.10','r2_network_100':'100-Mb/s stress'}
    for scenario,label in condlabels.items():
        for method in LABELS:
            r=summary[(summary.scenario==scenario)&(summary.method==method)].iloc[0]
            cells=[label,LABELS[method]]
            for metric,scale,digits in [('success_ratio',100,2),('task_energy_mj',1,3),('p95_latency_ms',.001,2),('jain_tenant_success',1,4),('minimum_tenant_success_ratio',100,2)]:
                cells.append('No service' if metric=='jain_tenant_success' and r.no_service else fmt(r[metric+'_mean'],r[metric+'_ci95'],digits,scale))
            lines.append(' & '.join(cells)+r'\\')
        r=summary[(summary.scenario==scenario+'_no_energy_guard')&(summary.method=='scd')].iloc[0]
        lines.append(' & '.join([label,'No energy guard']+[fmt(r[m+'_mean'],r[m+'_ci95'],d,s) for m,s,d in [('success_ratio',100,2),('task_energy_mj',1,3),('p95_latency_ms',.001,2),('jain_tenant_success',1,4),('minimum_tenant_success_ratio',100,2)]])+r'\\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule'
    (TEX/'r2_factorial_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (TEX/'r2_factorial_body.tex').write_text('\n'.join(lines[2:-1])+'\n',encoding='utf-8')
    compact=[r'Policy & Nominal success (\%) & Network success (\%) & Budget 0.30 success (\%) \\',r'\midrule']
    for method in ['scd','scd_no_debt','scd_no_debt_feedback','scd_no_forecast','scd_static_weights','no_energy_guard']:
        cells=[LABELS.get(method,'No energy guard')]
        for scenario in ['r2_nominal','r2_network_100','r2_energy_03']:
            r=summary[(summary.scenario==scenario+('_no_energy_guard' if method=='no_energy_guard' else ''))&(summary.method==('scd' if method=='no_energy_guard' else method))].iloc[0]
            cells.append(fmt(r.success_ratio_mean,r.success_ratio_ci95,2,100))
        compact.append(' & '.join(cells)+r'\\')
    compact.append(r'\bottomrule')
    (TEX/'r2_mechanism_table.tex').write_text('\n'.join(compact)+'\n',encoding='utf-8')
    energy=[r'Budget & Allocation & Success (\%) & Task (MJ) & Jain & Minimum tenant (\%) \\',r'\midrule']
    for scenario,label in [('r2_energy_03','0.30'),('r2_energy_01','0.10')]:
        for method in ['scd','scd_elastic_energy','scd_shared_energy']:
            r=summary[(summary.scenario==scenario)&(summary.method==method)].iloc[0]
            cells=[label,LABELS[method],fmt(r.success_ratio_mean,r.success_ratio_ci95,2,100),fmt(r.task_energy_mj_mean,r.task_energy_mj_ci95,3),
                   'No service' if r.no_service else fmt(r.jain_tenant_success_mean,r.jain_tenant_success_ci95,4),fmt(r.minimum_tenant_success_ratio_mean,r.minimum_tenant_success_ratio_ci95,2,100)]
            energy.append(' & '.join(cells)+r'\\')
    energy.append(r'\bottomrule')
    (TEX/'r2_allocation_table.tex').write_text('\n'.join(energy)+'\n',encoding='utf-8')
    claims={}
    for method,label in [('scd_no_debt_feedback','NoDebt'),('scd_no_debt','NoPrice')]:
        r=paired[(paired.scenario=='r2_nominal')&(paired.challenger==method)&(paired.reference=='scd')&(paired.metric=='success_ratio')].iloc[0]
        for suffix,value,digits in [('SuccessLossPP',-100*r.difference_mean,3),('SuccessLossPPCI',100*r.difference_ci95,3),('Holm',r.p_holm_global,5)]:
            claims['Rtwo'+label+suffix]=dict(value=float(value),formatted=f'{value:.{digits}f}',source='paired_effects.csv',scenario='r2_nominal',method=method,metric='success_ratio')
    for scenario,method,label in [('r2_nominal_no_energy_guard','scd','NoGuardNominal'),('r2_energy_03_no_energy_guard','scd','NoGuardModerate'),
        ('r2_energy_01','scd_elastic_energy','ElasticTight'),('r2_energy_01','scd_shared_energy','SharedTight'),
        ('r2_energy_03','scd','FixedModerate'),('r2_energy_03','scd_elastic_energy','ElasticModerate')]:
        r=summary[(summary.scenario==scenario)&(summary.method==method)].iloc[0]
        for metric,metriclabel,scale,digits in [('success_ratio','Success',100,2),('minimum_tenant_success_ratio','Minimum',100,2),('task_energy_mj','TaskMJ',1,3)]:
            for suffix,field in [('',metric+'_mean'),('CI',metric+'_ci95')]:
                value=float(r[field]*scale)
                claims['Rtwo'+label+metriclabel+suffix]=dict(value=value,formatted=f'{value:.{digits}f}',source='group_summary.csv',scenario=scenario,method=method,metric=metric)
    (TEX/'r2_component_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v['formatted']+'}' for k,v in claims.items())+'\n',encoding='utf-8')
    (OUT/'prose_claim_registry.json').write_text(json.dumps(claims,indent=2),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(dict(complete=True,runs=200,baseline_regressions=regressions,
        raw_result_sha256=raw_hashes,raw_records_match_csv=True,
        execution_source_sha256=protocol['execution_source_sha256'],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contrast_tests=len(contrasts),multiplicity='Holm across every prespecified metric contrast in all four conditions',
        source_sha256=hashlib.sha256((RUN/'campaign_results.csv').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(paired[(paired.metric=='success_ratio')&(paired.scenario.isin(['r2_nominal','r2_energy_01']))][['scenario','challenger','reference','difference_mean','difference_ci95','p_holm_global']].to_string(index=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=RUN)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--tex',type=Path,default=TEX)
    args=parser.parse_args()
    RUN=args.directory
    OUT=args.out or RUN/'analysis'
    BASELINE=args.baseline
    TEX=args.tex
    main()
