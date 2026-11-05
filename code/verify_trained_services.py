"""Rerun complete fixed-epoch training and verify the deployed CNN/RF artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

CODE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=CODE/'results/r2/trained_services_20261105')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    target=args.out.resolve()
    target.mkdir(parents=True,exist_ok=False)
    package=target/'execution/src'
    package.mkdir(parents=True)
    hashes={}
    for name in ['__init__.py','train_r2_workloads.py','r2_portable_models.py']:
        source=CODE/'src'/name
        shutil.copy2(source,package/name)
        hashes[name]=digest(source)
    registered=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        scope='Source snapshot registered before this verification rerun. This is not a retrospective claim of preregistering the initial training source.')
    (target/'verification_protocol.json').write_text(json.dumps(registered,indent=2),encoding='utf-8')
    env=os.environ.copy()
    env['PYTHONSAFEPATH']='1'
    env['PYTHONPATH']=str(package.parent)
    with (target/'training.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-P','-m','src.train_r2_workloads','--out',str(target/'trained')],
            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    reference=args.reference.resolve()
    comparisons=[]
    for name in ['training_losses.json','evaluation.json','cnn/model.json','forest/model.json']:
        original=reference/name
        generated=target/'trained'/name
        if original.read_bytes()!=generated.read_bytes():
            raise AssertionError('Deterministic full-training artifact mismatch: '+name)
        comparisons.append(dict(file=name,comparison='byte-identical',sha256=digest(generated)))
    old=json.loads((reference/'protocol.json').read_text())
    new=json.loads((target/'trained/protocol.json').read_text())
    old.pop('started_utc')
    new.pop('started_utc')
    if old!=new:
        raise AssertionError('Training data, split, hyperparameters or framework changed')
    for service in ['cnn','forest']:
        name=service+'/validation_inputs.npz'
        with np.load(reference/name,allow_pickle=False) as original,np.load(target/'trained'/name,allow_pickle=False) as generated:
            if set(original.files)!=set(generated.files):
                raise AssertionError('Training validation array keys changed')
            for key in original.files:
                np.testing.assert_array_equal(original[key],generated[key],strict=True)
            comparisons.append(dict(file=name,comparison='identical keys, dtypes, shapes and every array value'))
    if hashes!={name:digest(package/name) for name in hashes}:
        raise AssertionError('Verification source snapshot changed')
    report=dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        source_registration=registered['scope'],comparisons=comparisons,
        cnn_epochs=80,forest_trees=64,training_examples=1347,heldout_examples=450)
    (target/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,comparisons=len(comparisons),cnn_epochs=80,forest_trees=64)))


if __name__=='__main__':
    main()
