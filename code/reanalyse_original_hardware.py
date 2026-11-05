"""Recompute original physical evidence with immutable R1 analysis sources.

The path view redirects only explicitly named derived outputs. Every input
continues to come from the archived acquisition, which is never rewritten.
Run through reproduce.py so that the frozen source package has import priority.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent
HASH_EQUIVALENCES={}
METADATA_EQUIVALENCES=[]


def register_hashes(first,second):
    HASH_EQUIVALENCES.setdefault(first,set()).add(second)
    HASH_EQUIVALENCES.setdefault(second,set()).add(first)


def equivalent_hash(first,second):
    pending=[first]
    seen=set()
    while pending:
        value=pending.pop()
        if value==second:
            return True
        if value not in seen:
            seen.add(value)
            pending.extend(HASH_EQUIVALENCES.get(value,set())-seen)
    return False


class OutputView(os.PathLike):
    def __init__(self,source,destination,outputs):
        self.source=Path(source).resolve()
        self.destination=Path(destination).resolve()
        self.outputs=set(outputs)
        if self.destination==self.source or self.destination.is_relative_to(self.source):
            raise ValueError('Reanalysis outputs must be outside the original campaign')

    def __fspath__(self):
        return str(self.source)

    def __str__(self):
        return str(self.source)

    def __truediv__(self,name):
        return (self.destination if str(name) in self.outputs else self.source)/name


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def numeric_equal(left,right,path='root'):
    if isinstance(left,dict):
        if not isinstance(right,dict) or set(left)!=set(right):
            raise AssertionError('JSON keys differ at '+path)
        for key in left:
            numeric_equal(left[key],right[key],path+'.'+key)
    elif isinstance(left,list):
        if not isinstance(right,list) or len(left)!=len(right):
            raise AssertionError('JSON shape differs at '+path)
        for index,(a,b) in enumerate(zip(left,right)):
            numeric_equal(a,b,path+'['+str(index)+']')
    elif isinstance(left,str):
        if not isinstance(right,str):
            raise AssertionError('JSON type differs at '+path)
        if left!=right:
            if len(left)==64 and len(right)==64 and equivalent_hash(left,right):
                METADATA_EQUIVALENCES.append(dict(field=path,reason='Verified acquisition-to-portable/public hash chain'))
            else:
                raise AssertionError('JSON scientific label or unverified metadata differs at '+path)
    elif isinstance(left,bool) or left is None:
        if left!=right or type(left)!=type(right):
            raise AssertionError('JSON state differs at '+path)
    elif not np.isclose(left,right,rtol=1e-12,atol=1e-12,equal_nan=True):
        raise AssertionError('JSON numerical value differs at '+path)


def compare(left,right):
    if left.suffix=='.csv':
        pd.testing.assert_frame_equal(pd.read_csv(left),pd.read_csv(right),
            check_exact=False,rtol=1e-12,atol=1e-12)
    elif left.suffix=='.json':
        numeric_equal(json.loads(left.read_text(encoding='utf-8')),
                      json.loads(right.read_text(encoding='utf-8')))
    elif left.read_bytes()!=right.read_bytes():
        raise AssertionError('Generated evidence differs: '+left.name)
    register_hashes(digest(left),digest(right))
    return dict(file=left.name,reference_sha256=digest(left),recomputed_sha256=digest(right),
                numerical_match=True)


def public_collection_entry(path,entry,record):
    """Verify both the acquisition identity and the distributed file bytes."""
    actual=digest(path)
    size=path.stat().st_size
    if actual==entry['sha256'] and size==entry['bytes']:
        return dict(entry)
    if (record is None or record['original_sha256']!=entry['sha256']
            or record['source_bytes']!=entry['bytes']
            or record['public_sha256']!=actual or record['public_bytes']!=size
            or record['transformation']!='deployment_redaction_or_portable_path_serialization'):
        raise AssertionError('Unverified change to collected container evidence: '+path.name)
    return {**entry,'sha256':actual,'bytes':size}


def container_analysis_view(source,target):
    """Rebase validation metadata in a temporary copy, leaving all inputs intact.

    Frozen R1 checks acquisition byte counts and hashes directly. Redacted
    runtime-information records have different bytes. Verify their complete
    original/public provenance before supplying the frozen analyzer with an
    isolated collection-metadata view. Measurements and frozen code are never
    altered, and the distributed acquisition manifest is never overwritten.
    """
    artifact=CODE.parent
    provenance=artifact/'PUBLIC_PROVENANCE.json'
    if not provenance.is_file():
        return source
    rows={r['path']:r for r in json.loads(provenance.read_text(encoding='utf-8'))['files']}
    collections=[]
    for path in sorted(source.glob('*/runtime_evidence/collection.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        changed=[]
        copied=[]
        for entry in value['copied']:
            if Path(entry['local']).name!=entry['local']:
                raise AssertionError('Invalid collected evidence path')
            evidence=path.parent/entry['local']
            relative=evidence.relative_to(artifact).as_posix()
            updated=public_collection_entry(evidence,entry,rows.get(relative))
            copied.append(updated)
            if updated!=entry:
                changed.append(relative)
        collections.append((path,{**value,'copied':copied},changed))
    if not any(changed for _,_,changed in collections):
        return source
    view=target/'verified_public_input_view'
    shutil.copytree(source,view,ignore=shutil.ignore_patterns('analysis'))
    for original,value,changed in collections:
        (view/original.relative_to(source)).write_text(json.dumps(value,indent=2),encoding='utf-8')
        for name in changed:
            METADATA_EQUIVALENCES.append(dict(field=name,
                reason='Original collection hash/size and distributed public hash/size both verified before rebasing validation-only metadata in an isolated input view'))
    return view


def module(name):
    loaded=importlib.import_module('src.'+name)
    source=Path(loaded.__file__)
    expected=CODE/'frozen_r1_src'/source.name
    if digest(source)!=digest(expected):
        raise ValueError('Use the exact frozen R1 analysis source: '+name)
    return loaded


def call_main(name,**arguments):
    loaded=module(name)
    with patch.object(argparse.ArgumentParser,'parse_args',return_value=argparse.Namespace(**arguments)):
        loaded.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=args.out.resolve()
    if root.exists():
        raise ValueError('Use a new output directory for a fresh original-hardware reanalysis')
    root.mkdir(parents=True)
    bridge=CODE/'results/r2/original_provenance.json'
    if bridge.is_file():
        for row in json.loads(bridge.read_text())['files']:
            register_hashes(row['source_sha256'],row['archived_sha256'])
    public=CODE.parent/'PUBLIC_PROVENANCE.json'
    if public.is_file():
        for row in json.loads(public.read_text())['files']:
            register_hashes(row['original_sha256'],row['public_sha256'])
    physical=CODE/'results/revision/physical'
    comparisons=[]
    def finish(name,files):
        for relative in files:
            comparisons.append(dict(campaign=name,**compare(physical/name/relative,root/name/relative)))
        print(json.dumps(dict(campaign=name,verified_files=len(files))),flush=True)
    for name in ['inference_calibration_v2_20260905','inference_calibration_port04_repeat_20260909']:
        target=root/name
        target.mkdir()
        files=['timing_records.csv','service_models.json']
        module('summarize_inference_calibration').summarize(OutputView(physical/name,target,files))
        finish(name,files)
    name='activation_transport_isolated_20260905'
    target=root/name
    target.mkdir()
    files=['transport_timing_records.csv','activation_transport_analysis.json']
    call_main('summarize_activation_transport',directory=OutputView(physical/name,target,files))
    finish(name,files)
    name='power_thermal_20260905'
    target=root/name
    target.mkdir()
    files=['power_samples.csv','plateau_statistics.csv','thermal_samples.csv','analysis.json',
           'thermal_fit_source.csv','thermal_fit_analysis.json']
    view=OutputView(physical/name,target,files)
    module('summarize_power_thermal').analyze(view)
    call_main('fit_thermal_calibration',directory=view)
    finish(name,files)
    name='physical_scheduler_full_v3_complete_20260906'
    target=root/name
    (target/'analysis').mkdir(parents=True)
    call_main('audit_physical_power_sampling',campaign=physical/name,out=target/'analysis/power_sampling_audit.json')
    module('summarize_physical_trials').summarize(OutputView(physical/name,target,{'analysis'}))
    files=['analysis/'+p.name for p in (target/'analysis').iterdir() if p.is_file() and p.suffix in {'.csv','.tex'}]
    finish(name,files)
    name='container_startup_full_v6_complete_20260906'
    target=root/name
    target.mkdir()
    container_source=container_analysis_view(physical/name,target)
    call_main('summarize_container_trials',campaign=OutputView(container_source,target,{'analysis'}))
    finish(name,['analysis/summary.json','analysis/container_activation_table.tex'])
    audit=dict(complete=True,original_scheduler_trials=280,container_activations=160,
        analysis_source_policy='Each imported scientific analysis module is byte-verified against frozen_r1_src.',
        output_policy='Only declared derived output paths are redirected. Raw acquisitions are unchanged.',
        wrapper_sha256=digest(Path(__file__)),comparisons=comparisons,
        string_policy='All scientific labels match exactly. Changed provenance hashes require a verified original/archive/public chain.',
        metadata_equivalences=METADATA_EQUIVALENCES)
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,files=len(comparisons))),flush=True)


if __name__=='__main__':
    main()
