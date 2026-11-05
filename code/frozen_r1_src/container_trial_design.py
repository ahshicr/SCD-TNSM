"""Predeclared container experiment and explicit private-runtime boundaries.

Pure functions only: importing this module never contacts or changes a board.
The two deployment stacks may use different Python builds; their contrast is
not an identification of pure kernel isolation overhead.
"""
import json
from pathlib import Path, PurePosixPath
import random
import re

PORTS = (5, 9, 7, 8)
BLOCKS = 20
PULLS = 5
RESIDENT_CALLS = 20
DOCKER_VERSION = '29.8.0'
IMAGE_TAG = 'docker.io/library/python:3.11.16-slim-bookworm'
SEED = 20260908


def validate_root(root):
    if not re.fullmatch(r'/var/tmp/tnsm-container\.[A-Za-z0-9]{8,32}', root):
        raise ValueError('Not an explicit, dedicated container experiment root')
    return root


def validate_store(store):
    if store not in ['pull_%02d' % n for n in range(PULLS)]:
        raise ValueError('Unknown private image-store repetition')
    return store


def runtime_paths(root, store):
    root, store = validate_root(root), validate_store(store)
    base = root + '/' + store
    return dict(root=root, base=base, data=base+'/data', execution=base+'/exec',
                socket=base+'/docker.sock', pid=base+'/dockerd.pid',
                config=base+'/daemon.json', client=base+'/client',
                log=base+'/dockerd.log')


def daemon_config(root, store):
    p = runtime_paths(root, store)
    return {'data-root': p['data'], 'exec-root': p['execution'],
            'pidfile': p['pid'], 'hosts': ['unix://'+p['socket']],
            'group': 'root', 'bridge': 'none', 'iptables': False,
            'ip6tables': False, 'ip-forward': False, 'ip-masq': False,
            'userland-proxy': False, 'storage-driver': 'vfs',
            'exec-opts': ['native.cgroupdriver=cgroupfs'],
            'features': {'containerd-snapshotter': False, 'cdi': False},
            'max-concurrent-downloads': 1, 'max-download-attempts': 1,
            'log-driver': 'none', 'shutdown-timeout': 15}


def docker_prefix(root, store):
    p = runtime_paths(root, store)
    return [root+'/docker/docker', '--host', 'unix://'+p['socket'],
            '--config', p['client']]


def image_digest(reference):
    if not re.fullmatch(r'(?:docker.io/)?(?:library/)?python@sha256:[0-9a-f]{64}', reference):
        raise ValueError('Inference requires the inspected immutable Python image digest')
    return reference


def worker_command(root, store, digest, name, sensors, uid, gid):
    if not re.fullmatch(r'tnsm-p\d{2}-b\d{2}', name):
        raise ValueError('Unexpected study container identity')
    if not sensors or int(uid) <= 0 or int(gid) <= 0:
        raise ValueError('Identified CPU sensor and non-root user are required')
    cmd = docker_prefix(root, store) + ['run', '--rm', '-i', '--pull=never',
        '--name', name, '--label', 'tnsm.container-study='+root.rsplit('/', 1)[1],
        '--network=none', '--read-only', '--cap-drop=ALL',
        '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=512m',
        '--user', '%d:%d' % (uid, gid), '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--mount', 'type=bind,src='+root+'/worker,dst=/work,readonly']
    for sensor in sensors:
        source = sensor['real_path']
        zone = sensor['zone']
        if not re.fullmatch(r'thermal_zone\d+', zone):
            raise ValueError('Unexpected sensor identity')
        pure = PurePosixPath(source)
        if '..' in pure.parts or not source.startswith('/sys/devices/') or ',' in source:
            raise ValueError('Sensor mount must be a resolved sysfs thermal directory')
        if pure.name != zone:
            raise ValueError('Sensor path/identity mismatch')
        cmd += ['--mount', 'type=bind,src='+source+',dst=/sys/class/thermal/'+zone+',readonly']
    return cmd + [image_digest(digest), 'python3', '-B', '-u', '/work/board_inference_worker.py']


def design():
    rng = random.Random(SEED)
    rows = []
    for port in PORTS:
        for block in range(BLOCKS):
            methods = ['host_process', 'cached_container']
            rng.shuffle(methods)
            for order, method in enumerate(methods):
                rows.append(dict(port=port, block=block, within_block_order=order,
                    method=method, first_input=(block*7) % 128,
                    resident_inputs=[(block*7+i+1) % 128 for i in range(RESIDENT_CALLS)],
                    cooldown_seconds=10., temperature_guard_c=65.))
    return rows


def campaign_gate(directory):
    """Refuse activity unless all prior trials and their cleanup are confirmed."""
    directory = Path(directory)
    status_path = directory/'campaign_status.json'
    if not status_path.exists():
        raise RuntimeError('Policy campaign still active: no board operations or downloads')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    complete = status.get('completed', [])
    if not status.get('complete') or len(complete) != manifest['planned_trials']:
        raise RuntimeError('Previous full policy campaign is incomplete')
    if sorted(x['ordinal'] for x in complete) != list(range(manifest['planned_trials'])):
        raise RuntimeError('Policy trial identities are not a complete matrix')
    for entry in complete:
        summary = json.loads((directory/('trial_%03d' % entry['ordinal'])/'summary.json').read_text(encoding='utf-8'))
        if not entry['complete'] or not summary.get('complete') or summary.get('error'):
            raise RuntimeError('Policy trial did not finish cleanly')
        cleanup = summary.get('cleanup', {})
        if not cleanup or any(c.get('errors') or not c.get('stop_acknowledged') or
                              not c.get('directory_removed') for c in cleanup.values()):
            raise RuntimeError('Policy worker cleanup has not been confirmed')
    power = status.get('power_sampler') or {}
    if power.get('errors') or not power.get('thread_stopped'):
        raise RuntimeError('Policy power sampler cleanup has not been confirmed')
    return status


def archive_gate(campaign, archive):
    """Do not contaminate startup timing with the previous analysis or NAS copy."""
    campaign, archive = Path(campaign), Path(archive)
    report_path = archive/'archive_verification.json'
    analysis_path = campaign/'analysis'/'audit.json'
    if not report_path.exists() or not analysis_path.exists():
        raise RuntimeError('Previous statistics and byte-verified NAS archive are not finished')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    if not report.get('campaign_complete') or not report.get('analysis_audit_present') or not analysis.get('complete'):
        raise RuntimeError('Previous statistics/archive did not pass')
    if Path(report['source']).resolve() != campaign.resolve():
        raise RuntimeError('Archive belongs to a different policy campaign')
    if analysis.get('trials') != 280 or analysis.get('blocks') != 10 or analysis.get('parity_failures') != 0:
        raise RuntimeError('Previous formal protocol did not pass full-matrix verification')
    return report
