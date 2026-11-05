"""Twenty repeated calibrations per service and used batch on each live board."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from src.board_inference_client import BoardClient

PORTS = [1,3,4,5,7,8,9,10,12,17,19,21,22,23]

def load_services(original, added):
    locations = {'mlp': original, 'cnn': added/'cnn', 'forest': added/'forest'}
    models = {k:json.loads((v/'model.json').read_text(encoding='utf-8')) for k,v in locations.items()}
    examples = {k:dict(np.load(v/'validation_inputs.npz')) for k,v in locations.items()}
    return dict(kind='collection', services=models), examples

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--original',type=Path,default=Path('results/revision/physical/portable_model_20260905'))
    p.add_argument('--added',type=Path,default=Path('results/r2/trained_services_20261105'))
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    model,examples=load_services(a.original,a.added)
    rows=[r for r in json.loads(a.inventory.read_text(encoding='utf-8'))['ports'] if r['port'] in PORTS]
    if len(rows)!=len(PORTS):
        raise ValueError('Incomplete inventory')
    jobs=[(service,batch,repeat) for service in ['mlp','cnn','forest'] for batch in [1,4,16] for repeat in range(20)]
    random.Random(20261105).shuffle(jobs)
    names=['calibrate_r2_services.py','board_inference_client.py','board_inference_worker.py','portable_inference.py','r2_portable_models.py']
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),ports=PORTS,jobs_per_board=len(jobs),jobs=jobs,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
        model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
        timing='Remote compute timestamps, including per-inference thermal checks. Controller time is not used as transport calibration.',
        temperature_guard_c=65., rest='max(0.1 s, 1.5 times compute duration)',
        selection='All 20 repetitions for all three workloads and all three used batch sizes, no selection by speed')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    key=os.environ['EDGE_SSH_KEY']
    def measure(row):
        path=a.out/('port_%02d.jsonl'%row['port'])
        client=None
        values=[]
        report=dict(port=row['port'],complete=False)
        try:
            client=BoardClient(row,key,model)
            report['ready']=client.ready
            with path.open('x',encoding='utf-8') as f:
                for ordinal,(service,batch,repeat) in enumerate(jobs):
                    data=examples[service]
                    ix=(np.arange(batch)+7*repeat)%len(data['features'])
                    result=client.infer(ordinal,data['features'][ix],service_id=service)
                    expected=data['expected'][ix]
                    actual=np.asarray(result.get('predictions',[]))
                    parity=result.get('status')=='ok' and len(actual)==len(expected) and bool(np.all(np.abs(actual-expected)/np.maximum(np.abs(expected),1.)<1e-4))
                    result.update(service_id=service,batch=batch,repetition=repeat,parity_ok=parity)
                    f.write(json.dumps(result)+'\n')
                    f.flush()
                    if not parity:
                        raise RuntimeError('Calibration did not complete with output parity')
                    compute=result['compute_end']-result['compute_start']
                    values.append(dict(service=service,batch=batch,compute_s=compute))
                    time.sleep(max(.1,1.5*compute))
            report['service_models']={service:{str(batch):float(np.mean([v['compute_s'] for v in values if v['service']==service and v['batch']==batch])) for batch in [1,4,16]} for service in ['mlp','cnn','forest']}
            report['complete']=len(values)==len(jobs)
        except Exception as exc:
            report['error']=repr(exc)
        finally:
            if client:
                report['cleanup']=client.close()
        (a.out/('port_%02d_summary.json'%row['port'])).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'port':row['port'],'complete':report['complete'],'groups':len(values),'error':report.get('error')}),flush=True)
        return report
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=[future.result() for future in as_completed([pool.submit(measure,row) for row in rows])]
    report=dict(complete=all(r['complete'] for r in results),boards=results,groups=sum(len(jobs) for r in results if r['complete']))
    (a.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['complete']:
        raise SystemExit('Incomplete calibration. Retaining all observed evidence.')

if __name__=='__main__':
    main()
