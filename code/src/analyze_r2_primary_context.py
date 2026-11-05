"""Practical magnitudes from the immutable R1 primary records."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/strict/campaign_formula_v7_20260914/campaign_results.csv'
OUT=ROOT/'results/r2/primary_context'
TEX=ROOT.parent/'manuscript/sections/r2_primary_macros.tex'
def ci(values):
    values=np.asarray(values,float)
    return float(values.mean()),float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    TEX.parent.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE)
    data=data[data.scenario=='primary']
    rows=[]
    scd=data[data.method=='scd'].set_index('seed').sort_index()
    macro=[]
    for other,label in [('dpp_matched','DPP'),('compute_first','Compute')]:
        reference=data[data.method==other].set_index('seed').loc[scd.index]
        if not np.array_equal(scd.evaluated_requests,reference.evaluated_requests):
            raise AssertionError('Offered denominators differ')
        for field,name,scale in [('on_time_requests','ExtraOnTime',1),('p95_latency_ms','PNinetyFiveDifferenceMs',1),('mean_peak_aoi_ms','AoIDifferenceS',.001)]:
            difference=(scd[field]-reference[field])*scale
            mean,half=ci(difference)
            for seed,value in difference.items():
                rows.append(dict(contrast='SCD minus '+other,seed=int(seed),metric=field,difference=float(value)))
            decimals=0 if field=='on_time_requests' else 2
            macro.extend([r'\newcommand{\Rtwo'+label+name+'}{'+f'{mean:,.{decimals}f}'+'}',r'\newcommand{\Rtwo'+label+name+'CI}{'+f'{half:,.{decimals}f}'+'}'])
    pd.DataFrame(rows).to_csv(OUT/'paired_seed_differences.csv',index=False)
    TEX.write_text('\n'.join(macro)+'\n',encoding='utf-8')
    (OUT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),formula='Within-seed SCD minus comparator, five-seed mean and Student-t 95 percent interval',source_keys=data[['key','method','seed']].to_dict('records')),indent=2),encoding='utf-8')
    print(TEX.read_text())
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--tex',type=Path,default=TEX)
    args=parser.parse_args()
    SOURCE=args.source
    OUT=args.out
    TEX=args.tex
    main()
