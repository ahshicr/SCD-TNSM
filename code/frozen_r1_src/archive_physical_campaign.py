"""Archive a stopped campaign, never an actively growing hardware dataset."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('campaign',type=Path)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    status_path=args.campaign/'campaign_status.json'
    while not status_path.exists():
        if not args.wait:
            raise ValueError('Campaign has not stopped; refusing an inconsistent live archive')
        time.sleep(30)
    status=json.loads(status_path.read_text(encoding='utf-8'))
    if not status.get('power_sampler',{}).get('thread_stopped'):
        raise ValueError('Sampler cleanup not confirmed')
    # Wait for the separate statistics/certificate process, but preserve the raw
    # run even if statistics fail. An archive is not a submission-readiness claim.
    analysis=args.campaign/'analysis'/'audit.json'
    deadline=time.monotonic()+900
    while status['complete'] and not analysis.exists() and time.monotonic()<deadline:
        time.sleep(30)
    manifest=json.loads((args.campaign/'manifest.json').read_text(encoding='utf-8'))
    snapshot=args.campaign/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name,expected in manifest['source_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe source snapshot name')
        source=Path(__file__).with_name(name)
        if digest(source)!=expected:
            raise ValueError('Frozen source no longer matches '+name)
        archived_source=snapshot/name
        if archived_source.exists():
            if digest(archived_source)!=expected:
                raise ValueError('Existing frozen source snapshot differs: '+name)
        else:
            shutil.copy2(source,archived_source)
    if args.target.exists():
        raise ValueError('Archive target exists; refusing an overwrite')
    shutil.copytree(args.campaign,args.target)
    files=[]
    for source in sorted(args.campaign.rglob('*')):
        if not source.is_file():
            continue
        relative=source.relative_to(args.campaign)
        copy=args.target/relative
        original_hash=digest(source)
        if source.stat().st_size != copy.stat().st_size or original_hash != digest(copy):
            raise ValueError('Archive verification failed: '+str(relative))
        files.append(dict(path=relative.as_posix(),bytes=source.stat().st_size,sha256=original_hash))
    report=dict(archived_utc=datetime.now(timezone.utc).isoformat(),
        campaign_complete=status['complete'],analysis_audit_present=analysis.exists(),
        source=str(args.campaign),target=str(args.target),files=files,
        scope='Byte-verified internal archive, not a public release or a claim of full reviewer closure')
    (args.target/'archive_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
    print(json.dumps(dict(files=len(files),bytes=sum(f['bytes'] for f in files))))


if __name__=='__main__':
    main()
