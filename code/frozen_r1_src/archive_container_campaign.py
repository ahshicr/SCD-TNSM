"""Create a byte-verified immutable copy of a completed container campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    source = args.campaign.resolve()
    target = args.target.resolve()
    if source == target or source in target.parents:
        raise ValueError('Archive target must be outside the source campaign')
    if target.exists():
        raise ValueError('Archive target exists; refusing an overwrite')
    status = json.loads((source / 'campaign_status.json').read_text(encoding='utf-8'))
    audit = json.loads((source / 'analysis' / 'audit.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if not status.get('complete') or not audit.get('complete'):
        raise ValueError('Container collection or analysis is incomplete')
    if audit.get('activation_count') != manifest.get('planned_activations'):
        raise ValueError('Analysis/manifest activation count mismatch')
    for board in status.get('boards', []):
        if (not board.get('complete') or board.get('error') or
                board.get('cleanup_error') or board.get('evidence_error')):
            raise ValueError('A board lacks completed collection and cleanup evidence')
        for cleanup in board.get('cleanup', []):
            if (not cleanup.get('daemon_stopped') or cleanup.get('remaining_mounts') or
                    cleanup.get('remaining_store_processes') or
                    not cleanup.get('forwarding_unchanged')):
                raise ValueError('A private runtime was not cleanly stopped')
    shutil.copytree(source, target)
    files = []
    for original in sorted(source.rglob('*')):
        if not original.is_file():
            continue
        relative = original.relative_to(source)
        copied = target / relative
        checksum = digest(original)
        if original.stat().st_size != copied.stat().st_size or checksum != digest(copied):
            raise ValueError('Archive verification failed: ' + relative.as_posix())
        files.append(dict(path=relative.as_posix(), bytes=original.stat().st_size,
                          sha256=checksum))
    report = dict(
        archived_utc=datetime.now(timezone.utc).isoformat(),
        campaign_complete=True,
        analysis_audit_present=True,
        source=str(source),
        target=str(target),
        files=files,
        scope=('Byte-verified internal archive; failed preliminary campaigns remain '
               'separate and are not pooled with this protocol.'))
    (target / 'archive_verification.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
    print(json.dumps(dict(files=len(files), bytes=sum(f['bytes'] for f in files))), flush=True)


if __name__ == '__main__':
    main()
