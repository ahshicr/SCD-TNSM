"""Regenerate the three-service calibration table from every completed group."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t

def main(directory,tex,out=None):
    protocol=json.loads((directory/'protocol.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['complete'] or summary['groups']!=2520 or len(summary['boards'])!=14:
        raise ValueError('Incomplete three-service calibration')
    for name,digest in protocol['source_sha256'].items():
        if hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Calibration source snapshot changed')
    raw=[]
    expected={(s,b,r) for s in ['mlp','cnn','forest'] for b in [1,4,16] for r in range(20)}
    for board in summary['boards']:
        port=board['port']
        records=[json.loads(s) for s in (directory/('port_%02d.jsonl'%port)).read_text().splitlines()]
        identities={(r['service_id'],r['batch'],r['repetition']) for r in records}
        if identities!=expected or len(records)!=180:
            raise ValueError('Missing or duplicate service calibration group')
        for r in records:
            if r['status']!='ok' or not r['parity_ok'] or len(r['predictions'])!=r['batch']:
                raise ValueError('Invalid prediction group')
            raw.append(dict(port=port,board=board['ready']['board'],architecture=board['ready']['arch'],
                service=r['service_id'],batch=r['batch'],repetition=r['repetition'],
                service_ms=1000*(r['compute_end']-r['compute_start']),
                before_cpu_c=r['before']['cpu_c'],after_cpu_c=r['after']['cpu_c']))
    data=pd.DataFrame(raw)
    out=out or directory/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'service_timing_records.csv',index=False)
    rows=[]
    for (port,board,service,batch),g in data.groupby(['port','board','service','batch']):
        if len(g)!=20:
            raise ValueError('Twenty calibration repetitions required')
        values=g.service_ms.to_numpy()
        rows.append(dict(port=port,board=board,service=service,batch=batch,n=20,
            mean_ms=float(values.mean()),ci95_ms=float(t.ppf(.975,19)*values.std(ddof=1)/np.sqrt(20)),
            minimum_cpu_c=float(min(g.before_cpu_c.min(),g.after_cpu_c.min())),maximum_cpu_c=float(max(g.before_cpu_c.max(),g.after_cpu_c.max()))))
    grouped=pd.DataFrame(rows)
    grouped.to_csv(out/'service_summary.csv',index=False)
    lines=[r'Port & Board & MLP (ms) & CNN (ms) & Forest (ms)\\',r'\midrule']
    for port,g in grouped[grouped.batch==16].groupby('port'):
        name=str(g.board.iloc[0])
        name=('Pi 5' if 'Pi 5' in name else 'Pi 4' if 'Pi 4' in name else 'Pi 3' if 'Pi 3' in name else 'Pi 2' if 'Pi 2' in name else 'RDK X5')
        cells=[str(port),name]
        for service in ['mlp','cnn','forest']:
            r=g[g.service==service].iloc[0]
            cells.append('$%.2f\\pm%.2f$'%(r.mean_ms,r.ci95_ms))
        lines.append(' & '.join(cells)+r'\\')
    lines.append(r'\bottomrule')
    tex.mkdir(parents=True,exist_ok=True)
    (tex/'r2_service_calibration_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    digest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [directory/'protocol.json',directory/'summary.json',*directory.glob('port_*.jsonl')]}
    report=dict(complete=True,boards=14,services=3,batch_sizes=[1,4,16],repetitions_per_cell=20,groups=len(data),
        individual_inferences=int(data.batch.sum()),parity_failures=0,source_sha256=digest,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervals='Student t across 20 calibration repetitions within each board/service/batch cell, not independent deployments',
        service_scope='Complete trained model execution in portable Python, including thermal checks. CPU-only timings for every platform.')
    (out/'source_snapshot').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--tex',type=Path,default=Path(__file__).resolve().parents[2]/'manuscript/sections')
    a=p.parse_args()
    main(a.directory,a.tex,a.out)
