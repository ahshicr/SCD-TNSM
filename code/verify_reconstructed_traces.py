"""Compare regenerated trace bytes and semantics with the archived R1 manifests."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def semantics(manifest):
    result=deepcopy(manifest)
    # An input's location changes on another computer. Its bytes do not.
    result['source'].pop('path',None)
    return result


def verify(trace_root,reference,required_registry=None):
    archived={p.parent.name:p for p in reference.glob('*/manifest.json')}
    generated={p.parent.name:p for p in trace_root.glob('*/manifest.json')}
    if not archived or not generated or set(generated)-set(archived):
        raise ValueError('Empty or unregistered reconstructed trace set')
    required=set(archived)
    if required_registry:
        registry=json.loads((trace_root/required_registry).read_text())
        required={Path(row['directory']).name for row in registry['traces']}
    if not required or not required<=set(generated):
        raise ValueError('Required reconstructed traces are missing')
    records=[]
    for name,path in sorted(generated.items()):
        actual=json.loads(path.read_text())
        expected=json.loads(archived[name].read_text())
        if semantics(actual)!=semantics(expected):
            raise ValueError('Trace differs from archived scientific metadata or array hashes: '+name)
        arrays=[]
        for key,item in sorted(expected['arrays'].items()):
            array=(path.parent/item['file']).resolve()
            if not array.is_relative_to(path.parent.resolve()):
                raise ValueError('Array path escapes trace directory')
            observed=digest(array)
            if observed!=item['sha256']:
                raise ValueError('Reconstructed array bytes differ from archived evidence: '+name+'/'+key)
            arrays.append(dict(array=key,sha256=observed,bytes=array.stat().st_size))
        records.append(dict(trace=name,archived_manifest_sha256=digest(archived[name]),
            reconstructed_manifest_sha256=digest(path),requests=actual['sampling']['retained_requests'],arrays=arrays))
    return dict(complete=True,traces=len(records),required_traces=len(required),records=records,
        comparison='All scientific metadata and all reconstructed array bytes match archived R1 manifests. Only source.path is location-dependent.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--required-registry')
    args=parser.parse_args()
    report=verify(args.trace_root,args.reference,args.required_registry)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['complete','traces','required_traces']}))


if __name__=='__main__':
    main()
