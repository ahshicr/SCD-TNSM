"""Rebuild every retained manuscript figure with its frozen R1 source.

Numerical plot inputs are compared at 1e-12. PDF pixels are compared exactly
after rendering both files with the same PyMuPDF version. PDF timestamps and
object identifiers are consequently irrelevant to the visual comparison.
Run through reproduce.py to select the immutable scientific source package.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import fitz
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_pdf(reference,current):
    with fitz.open(reference) as first,fitz.open(current) as second:
        if len(first)!=len(second):
            raise AssertionError('Figure page count changed: '+reference.name)
        records=[]
        for index,(a,b) in enumerate(zip(first,second)):
            if a.rect!=b.rect or a.get_text()!=b.get_text():
                raise AssertionError('Figure size or visible text changed: '+reference.name)
            left=a.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            right=b.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            if (left.width,left.height,left.n)!=(right.width,right.height,right.n):
                raise AssertionError('Figure raster shape changed: '+reference.name)
            if left.samples!=right.samples:
                x=np.frombuffer(left.samples,dtype=np.uint8).astype(np.int16)
                y=np.frombuffer(right.samples,dtype=np.uint8).astype(np.int16)
                raise AssertionError('Figure rendering changed: '+reference.name+
                    ' mean absolute channel difference '+str(float(np.abs(x-y).mean())))
            records.append(dict(page=index+1,render_sha256=hashlib.sha256(left.samples).hexdigest(),
                width=left.width,height=left.height,identical=True))
    return dict(file=reference.name,reference_sha256=sha(reference),recomputed_sha256=sha(current),pages=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    output=args.out.resolve()
    if output.exists():
        raise ValueError('Use a fresh figure verification directory')
    output.mkdir(parents=True)
    sources={}
    def run(module,arguments):
        loaded=importlib.import_module('src.'+module)
        source=Path(loaded.__file__)
        if sha(source)!=sha(CODE/'frozen_r1_src'/source.name):
            raise ValueError('Figure source differs from frozen R1: '+module)
        sources[module]=sha(source)
        env=os.environ.copy()
        env['MPLBACKEND']='Agg'
        with (output/(module+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-P','-m','src.'+module,*map(str,arguments)],
                cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    strict=CODE/'results/strict'
    arguments=[]
    for flag,name in [('campaign','campaign_formula_v7_20260914'),
        ('network','network_stress_formula_v4_20260914'),
        ('deadline','deadline_guard_network_stress_v4_20260914'),
        ('load','load_scaling_formula_v2_20260914'),
        ('breakdown','breakdown_formula_v1_20260914')]:
        arguments+=['--'+flag,strict/name]
    run('strict_paper_figures',[*arguments,'--out',output/'simulation'])
    physical=CODE/'results/revision/physical'
    run('physical_scheduler_figure',['--analysis',physical/'physical_scheduler_full_v3_complete_20260906/analysis',
        '--out',output/'physical/fig_physical_scheduler'])
    run('physical_calibration_figure',['--data',physical/'power_thermal_20260905',
        '--out',output/'physical','--final'])
    run('container_startup_figure',[physical/'container_startup_full_v6_complete_20260906',
        '--output-dir',output/'physical'])
    comparisons=[]
    plots=[]
    for folder,reference in [(output/'simulation',strict/'paper_figures_formula_v5_20260915'),
                              (output/'physical',CODE.parent/'manuscript/figures')]:
        for path in sorted(folder.glob('*.csv')):
            original=reference/path.name
            pd.testing.assert_frame_equal(pd.read_csv(original),pd.read_csv(path),
                check_exact=False,rtol=1e-12,atol=1e-12)
            comparisons.append(dict(file=path.name,reference_sha256=sha(original),
                recomputed_sha256=sha(path),numerical_match=True))
        for path in sorted(folder.glob('*.pdf')):
            original=CODE.parent/'manuscript/figures'/path.name
            plots.append(compare_pdf(original,path))
    if len(comparisons)!=16 or len(plots)!=9:
        raise AssertionError('The complete nine-figure, sixteen-source-table contract changed')
    report=dict(complete=True,figures=len(plots),source_tables=len(comparisons),
        script_sha256=sha(Path(__file__)),frozen_sources=sources,
        numerical_comparisons=comparisons,visual_comparisons=plots,
        scope='All original main and supplementary figures, regenerated without changing published evidence.')
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,figures=len(plots),source_tables=len(comparisons))),flush=True)


if __name__=='__main__':
    main()
