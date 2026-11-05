"""Preserve application-coherent tenants when constructing physical arrivals."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from src.strict_trace import _classify_functions, _tenant_from_application
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/strict/processed/workload_strict_v2.npz'
OUT=ROOT/'results/r2/workload_application_16x4.npz'
def main():
    data=np.load(SOURCE)
    counts=data['counts']
    strata,edges,memory_edges=_classify_functions(data['p50_ms'],data['memory_mb'])
    tenants=_tenant_from_application(data['app_ids'])
    aggregated=np.zeros((counts.shape[1],16,4),dtype=np.uint64)
    for function in range(len(tenants)):
        aggregated[:,int(strata[function]),int(tenants[function])]+=counts[function].astype(np.uint64)
    if int(aggregated.sum()) != 12444637039 or not np.array_equal(aggregated.sum(axis=(1,2)),counts.sum(axis=0)):
        raise AssertionError('Aggregation lost requests')
    np.savez_compressed(OUT,count=aggregated,tenant_names=np.asarray(['logical-tenant-'+str(k) for k in range(4)]),
                        duration_edges=edges,memory_edges=memory_edges)
    report=dict(utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE.relative_to(ROOT)),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grouping_source_sha256=hashlib.sha256(Path(__file__).with_name('strict_trace.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
        counts_total=int(aggregated.sum()),per_tenant_totals=[int(v) for v in aggregated.sum(axis=(0,1))],
        tenant_rule='Anonymized application ID modulo 4, identical to strict_trace._tenant_from_application',
        function_identity_scope='Aggregation only specifies the release-time/tenant/class distribution of new physical jobs. It is not used by the raw-function simulator or its caches.',
        conservation='Exact total and exact minute totals retained',
        simulation_separation='The 200 completed full-trace runs are unchanged and are not rerun or reweighted.')
    OUT.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    main()
