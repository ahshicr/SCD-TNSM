"""Reproduce the R1 and R2 matrices with their own immutable execution sources.

Run from this code directory. Fresh results are always separated from the
archived paper evidence. No command in this entry point contacts lab devices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE=Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_sources(destination: Path, base: Path, overlay: Path | None=None) -> Path:
    """Make importable snapshots, retaining exact source bytes and local hashes."""
    destination.mkdir(parents=True,exist_ok=True)
    sources={p.name:p for p in base.glob('*.py')}
    if overlay:
        sources.update({p.name:p for p in overlay.glob('*.py')})
    if '__init__.py' not in sources:
        raise ValueError('Execution source package is incomplete')
    target=destination/'src'
    target.mkdir(exist_ok=True)
    unexpected={p.name for p in target.glob('*.py')}-set(sources)
    if unexpected:
        raise ValueError('Refusing to mix source versions with stale modules: '+str(sorted(unexpected)))
    recorded={}
    for name,path in sorted(sources.items()):
        sha=digest(path)
        to=target/name
        if to.exists() and digest(to)!=sha:
            raise ValueError('Refusing to mix source versions in '+str(destination))
        if not to.exists():
            shutil.copy2(path,to)
        recorded[name]=sha
    (destination/'sources.json').write_text(json.dumps(recorded,indent=2),encoding='utf-8')
    return destination.resolve()


def run(arguments: list[str], source_root: Path | None=None) -> None:
    env=os.environ.copy()
    # -P removes the current-directory src package from implicit precedence.
    # Subprocesses created by the archived runner inherit this same behavior.
    env['PYTHONSAFEPATH']='1'
    env['PYTHONPATH']=str(source_root or CODE)
    command=[sys.executable,'-P',*arguments]
    print('RUN '+subprocess.list2cmdline(command),flush=True)
    subprocess.run(command,cwd=CODE,env=env,check=True)

def prepare_traces(root: Path,workers: int,primary_only: bool=False) -> Path:
    sources=package_sources(root/'execution_r1',CODE/'frozen_r1_src')
    traces=root/'traces'
    common=['--workload',str(CODE/'results/strict/processed/workload_strict_v2.npz'),
            '--trace-root',str(traces)]
    run(['-m','src.prepare_primary_traces',*common,'--workers',str(workers)],sources)
    if not primary_only:
        run(['-m','src.prepare_load_scaling_traces',*common],sources)
        run(['-m','src.prepare_breakdown_traces',*common,'--workers',str(workers)],sources)
    run([str(CODE/'verify_reconstructed_traces.py'),'--trace-root',str(traces),
         '--reference',str(CODE/'results/strict/trace_manifests_v3'),
         '--out',str(root/('primary_trace_verification.json' if primary_only else 'trace_verification.json')),
         *(['--required-registry','primary_trace_registry.json'] if primary_only else [])])
    return traces


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['baseline','additions','all','statistics','tests','upstream','traces','figures','conformance'])
    parser.add_argument('--output',type=Path,default=CODE/'results/reproduced_r2')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:
        raise ValueError('At least one worker is required')
    root=args.output.resolve()
    if root in [CODE,CODE/'results'] or root==CODE/'results/strict' or root==CODE/'results/r2':
        raise ValueError('Use a separate reproduction output directory')
    root.mkdir(parents=True,exist_ok=True)
    if args.mode=='traces':
        prepare_traces(root,args.workers)
    if args.mode in ['figures','all']:
        sources=package_sources(root/'execution_r1',CODE/'frozen_r1_src')
        run([str(CODE/'verify_reproduced_figures.py'),'--out',str(root/'figure_verification')],sources)
    if args.mode=='upstream':
        # This optional full-source step extracts the public archive and selects
        # the fixed top-9000 cohort. Existing processed evidence is not replaced.
        run([str(CODE/'frozen_input_src/preprocess.py'),'--archive',str(CODE/'data/azurefunctions_dataset2019.tar.xz'),
             '--work-dir',str(root/'upstream_raw'),'--out-dir',str(root/'upstream_processed'),'--functions','9000'])
        run(['-m','src.build_strict_workload','--workload',str(root/'upstream_processed/workload.npz'),
             '--archive',str(CODE/'data/azurefunctions_dataset2019.tar.xz'),'--out',str(root/'upstream_processed/workload_strict_v2.npz')])
    if args.mode in ['baseline','all']:
        traces=prepare_traces(root,args.workers)
        sources=package_sources(root/'execution_r1',CODE/'frozen_r1_src')
        run([str(CODE/'run_revision.py'),'--workers',str(args.workers),
             '--trace-root',str(traces),'--skip-traces',
             '--fresh-output-root',str(root/'baseline'),'--skip-tests'],sources)
    if args.mode in ['additions','all']:
        traces=root/'traces' if args.mode=='all' else prepare_traces(root,args.workers,primary_only=True)
        sources=package_sources(root/'execution_r2',CODE/'src',CODE/'results/r2/simulation_factorial_20261105/source_snapshot')
        run(['-m','src.run_r2_simulations','--trace-root',str(traces),
             '--out',str(root/'additions'),'--workers',str(args.workers)],sources)
        run(['-m','src.analyze_r2_simulations','--directory',str(root/'additions'),
             '--out',str(root/'statistics/additions'),'--tex',str(root/'tables')])
    if args.mode in ['statistics','all']:
        sources=package_sources(root/'execution_r1',CODE/'frozen_r1_src')
        campaigns=['campaign_formula_v7_20260914','network_stress_formula_v4_20260914',
                   'load_scaling_formula_v2_20260914','deadline_guard_network_stress_v4_20260914',
                   'breakdown_formula_v1_20260914','measured_startup_strict_v1_20260914']
        for name in campaigns:
            run(['-m','src.summarize_strict_campaign','--campaign',str(CODE/'results/strict'/name),
                 '--out',str(root/'statistics'/name)],sources)
        run([str(CODE/'reanalyse_original_hardware.py'),
             '--out',str(root/'statistics/original_hardware')],sources)
        run(['-m','src.analyze_r2_simulations','--out',str(root/'statistics/factorial'),'--tex',str(root/'tables')])
        run(['-m','src.analyze_r2_primary_context','--out',str(root/'statistics/primary'),
             '--tex',str(root/'tables/r2_primary_macros.tex')])
        # Reanalysis preserves the archived acquisition and analysis records.
        # Fresh reports and their own source snapshots use separate paths.
        for module,directory in [
            ('analyze_r2_calibration','service_calibration'),
            ('analyze_r2_physical','matched_physical_v4'),
            ('analyze_r2_switch','switch_contention_v3')]:
            run(['-m','src.'+module,'--directory',str(CODE/'results/r2'/directory),
                 '--out',str(root/'statistics'/directory),'--tex',str(root/'tables')])
        run(['-m','src.analyze_r2_interruption_sensitivity',
             '--formal',str(CODE/'results/r2/matched_physical_v4'),
             '--interruptions',str(CODE/'results/r2/acquisition_interruption_20261105'),
             '--out',str(root/'statistics/interruption_sensitivity')])
    if args.mode in ['tests','all']:
        run(['-m','pytest','-q','--junitxml',str(root/'tests.xml')])
    if args.mode in ['conformance','all']:
        traces=root/'traces'
        if not (traces/'binomial_p0004_seed20260914/release_s.npy').is_file():
            raise ValueError('Run the complete traces mode into this output directory before conformance')
        originals=['campaign_formula_v7_20260914','network_stress_formula_v4_20260914',
            'load_scaling_formula_v2_20260914','deadline_guard_network_stress_v4_20260914',
            'breakdown_formula_v1_20260914','measured_startup_strict_v1_20260914']
        for label,base,overlay,protocols in [
            ('r1',CODE/'frozen_r1_src',None,[CODE/'results/strict'/n/'protocol.json' for n in originals]),
            ('r2',CODE/'src',CODE/'results/r2/simulation_factorial_20261105/source_snapshot',
                [CODE/'results/r2/simulation_factorial_20261105/protocol.json'])]:
            sources=package_sources(root/('execution_'+label),base,overlay)
            shutil.copy2(CODE/'IMPLEMENTATION_SCOPE.md',sources/'IMPLEMENTATION_SCOPE.md')
            command=['-m','src.audit_implementation_conformance','--trace-root',str(traces),
                '--deployment',str(CODE/'config/strict_deployment_20260914.json'),
                '--exact-report',str(CODE/'results/strict/exact_benchmark_suite_v2_20260914/exact_benchmark_summary.json'),
                '--output',str(root/('conformance_'+label+'.json'))]
            for protocol in protocols:
                command+=['--protocol',str(protocol)]
            run(command,sources)


if __name__=='__main__':
    main()
