"""Run full statistical reanalysis and compare archived numerical evidence."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGNS=['campaign_formula_v7_20260914','network_stress_formula_v4_20260914',
    'load_scaling_formula_v2_20260914','deadline_guard_network_stress_v4_20260914',
    'breakdown_formula_v1_20260914','measured_startup_strict_v1_20260914']


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()

def compare_flattened_campaign(directory,reported):
    """Check all reported columns, including expanded nested scenario fields."""
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((directory/'runs').glob('*.json'))]
    flattened=pd.json_normalize(records)
    if flattened.key.duplicated().any() or len(flattened)!=len(reported):
        raise AssertionError('Duplicate or missing original run records')
    # Apply the original CSV serialization to list cells and nullable columns.
    flattened=pd.read_csv(io.StringIO(flattened.to_csv(index=False)))
    flattened=flattened.sort_values('key').reset_index(drop=True)
    reported=reported.sort_values('key').reset_index(drop=True)
    if set(flattened.columns)!=set(reported.columns):
        raise AssertionError('Original flattened campaign schema changed')
    pd.testing.assert_frame_equal(flattened[reported.columns],reported,
        check_exact=False,rtol=1e-12,atol=1e-12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    artifact=args.artifact.resolve()
    code=artifact/'code'
    workload_protocol=json.loads((code/'results/strict/processed/workload_strict_v2.json').read_text(encoding='utf-8'))
    upstream_archive=code/'data/azurefunctions_dataset2019.tar.xz'
    upstream_hash=sha(upstream_archive)
    if upstream_hash!=workload_protocol['source_archive_sha256']:
        raise AssertionError('The distributed Azure source archive differs from the workload protocol')
    output=args.output.resolve()
    if output.exists():
        raise ValueError('Choose a fresh reanalysis output path')
    if output.is_relative_to(artifact) and artifact!=ROOT:
        raise ValueError('Keep validation outputs outside the public artifact')
    output.mkdir(parents=True)
    env=os.environ.copy()
    env['PYTHONIOENCODING']='utf-8'
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTEST_ADDOPTS']='-p no:cacheprovider'
    env['NUMBA_CACHE_DIR']=str(output/'numba_cache')
    env['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    def run(name,command,source=None):
        local_env=env.copy()
        if source:
            local_env['PYTHONSAFEPATH']='1'
            local_env['PYTHONPATH']=str(source)
        with (output/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*command],cwd=code,env=local_env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Reanalysis failed at '+name+'. Inspect the private audit log.')
        print(json.dumps(dict(step=name,complete=True)),flush=True)
    run('statistics',['reproduce.py','statistics','--output',str(output/'reproduction')])
    reproduction=output/'reproduction'
    comparisons=[]
    def compare(reference,current):
        if reference.suffix=='.csv':
            pd.testing.assert_frame_equal(pd.read_csv(reference),pd.read_csv(current),
                check_exact=False,rtol=1e-12,atol=1e-12)
            mode='all rows, columns, labels and numerical values within 1e-12'
        else:
            if reference.read_bytes()!=current.read_bytes():
                raise AssertionError('Regenerated evidence changed: '+reference.name)
            mode='byte-identical'
        comparisons.append(dict(reference=reference.relative_to(artifact).as_posix(),
            output=current.relative_to(output).as_posix(),reference_sha256=sha(reference),
            output_sha256=sha(current),comparison=mode,passed=True))
    total=0
    for name in CAMPAIGNS:
        reference=code/'results/strict'/name
        current=reproduction/'statistics'/name
        audit=json.loads((current/'audit_and_summary.json').read_text(encoding='utf-8'))
        if audit['status']!='PASS':
            raise AssertionError('Original campaign conservation/provenance audit failed')
        for file in ['verified_runs.csv','reported_runs.csv','descriptive_summary.csv','paired_contrasts.csv']:
            compare(reference/'analysis'/file,current/file)
        # The generated verified records are loaded from every raw run JSON.
        verified=pd.read_csv(current/'verified_runs.csv').sort_values('key').reset_index(drop=True)
        reported=pd.read_csv(reference/'campaign_results.csv').sort_values('key').reset_index(drop=True)
        compare_flattened_campaign(reference,reported)
        total+=len(verified)
    if total!=1128:
        raise AssertionError('Original matrix count changed')
    strict=code/'results/strict'
    table_args=['-P','-m','src.strict_paper_tables']
    for flag,name in [('campaign',CAMPAIGNS[0]),('network',CAMPAIGNS[1]),('load',CAMPAIGNS[2]),
                      ('deadline',CAMPAIGNS[3]),('breakdown',CAMPAIGNS[4]),('measured-startup',CAMPAIGNS[5]),
                      ('exact','exact_benchmark_suite_v2_20260914')]:
        table_args+=['--'+flag,str(strict/name)]
    table_args+=['--deployment',str(code/'config/strict_deployment_20260914.json'),
                 '--out',str(reproduction/'tables_original')]
    run('original_tables',table_args,reproduction/'execution_r1')
    for path in sorted((strict/'paper_tables_formula_v3_20260914').glob('*.tex')):
        compare(path,reproduction/'tables_original'/path.name)
    for canonical,fresh in [
        ('simulation_factorial_20261105/analysis','factorial'),('primary_context','primary'),
        ('service_calibration/analysis','service_calibration'),
        ('matched_physical_v4/analysis','matched_physical_v4'),
        ('switch_contention_v3/analysis','switch_contention_v3'),
        ('interruption_sensitivity','interruption_sensitivity')]:
        reference_tables=sorted((code/'results/r2'/canonical).glob('*.csv'))
        if not reference_tables:
            raise AssertionError('Missing complete reference tables: '+canonical)
        for path in reference_tables:
            compare(path,reproduction/'statistics'/fresh/path.name)
    for path in sorted((reproduction/'tables').glob('*.tex')):
        compare(artifact/'manuscript/sections'/path.name,path)
    hardware=json.loads((reproduction/'statistics/original_hardware/audit.json').read_text(encoding='utf-8'))
    if not hardware['complete']:
        raise AssertionError('Original raw hardware reanalysis incomplete')
    run('figures',['reproduce.py','figures','--output',str(reproduction)])
    figures=json.loads((reproduction/'figure_verification/audit.json').read_text(encoding='utf-8'))
    if not figures['complete'] or figures['figures']!=9 or figures['source_tables']!=16:
        raise AssertionError('The complete original figure regeneration did not match')
    run('tests',['reproduce.py','tests','--output',str(reproduction)])
    xml=ET.parse(reproduction/'tests.xml')
    suites=list(xml.getroot().iter('testsuite'))
    test_totals={key:sum(int(s.get(key,'0')) for s in suites) for key in ['tests','failures','errors','skipped']}
    if test_totals['failures'] or test_totals['errors'] or not test_totals['tests']:
        raise AssertionError('Full test suite did not pass')
    run('trace_reconstruction',['reproduce.py','traces','--output',str(reproduction),'--workers','5'])
    traces=json.loads((reproduction/'trace_verification.json').read_text(encoding='utf-8'))
    if not traces['complete'] or traces['traces']!=81 or traces['required_traces']!=81:
        raise AssertionError('Full request-array reconstruction did not match the archived evidence')
    run('conformance',['reproduce.py','conformance','--output',str(reproduction)])
    conformance={}
    for label in ['r1','r2']:
        path=reproduction/('conformance_'+label+'.json')
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['status']!='PASS' or record['failed'] or record['checks']!=record['passed']:
            raise AssertionError('Executable model conformance failed: '+label)
        conformance[label]={k:record[k] for k in ['status','checks','passed','failed']}
        conformance[label]['report_sha256']=sha(path)
    run('full_model_training',['verify_trained_services.py','--out',str(reproduction/'training_verification')])
    training=json.loads((reproduction/'training_verification/audit.json').read_text(encoding='utf-8'))
    if not training['complete']:
        raise AssertionError('Complete training did not reproduce deployed models')
    report=dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),
        original_simulations=1128,added_simulations=200,
        upstream_archive=dict(sha256=upstream_hash,bytes=upstream_archive.stat().st_size,
            producer_asset='Azure/AzurePublicDataset release dataset-functions-2019'),
        original_hardware=hardware,new_calibration_groups=2520,new_scheduler_trials=140,new_switch_trials=60,
        tests=test_totals,comparisons=comparisons,figure_verification=figures,trace_reconstruction=traces,
        executable_conformance=conformance,full_training_verification=training,
        evidence_policy='Full archived result and raw-hardware reanalysis with source-isolated R1 analyses. No live hardware rerun or partial matrix is substituted.',
        script_sha256=sha(Path(__file__)))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,comparisons=len(comparisons),tests=test_totals)),flush=True)


if __name__=='__main__':
    main()
