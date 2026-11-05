"""Dedicated, non-service Docker runtime helper, invoked through existing sudo.

No operation at import. Never uses a default Docker socket or deletes a tree.
Generated private image stores are retained until evidence is copied and their
explicit cleanup is separately verified. This file is also uploaded to a Pi.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request

try:
    from container_trial_design import (DOCKER_VERSION, daemon_config, docker_prefix,
                                        image_digest, runtime_paths, validate_root)
except ImportError:
    from src.container_trial_design import (DOCKER_VERSION, daemon_config, docker_prefix,
                                            image_digest, runtime_paths, validate_root)


def read(path):
    try:
        return Path(path).read_text().strip().replace('\x00', '')
    except (OSError, UnicodeError):
        return None


def utc():
    return datetime.now(timezone.utc).isoformat()


def root_path(root):
    validate_root(root)
    path = Path(root)
    if path.is_symlink() or not path.is_dir() or str(path.resolve()) != root:
        raise ValueError('Runtime root must be an existing, resolved non-symlink directory')
    marker = json.loads((path/'owner.json').read_text())
    if marker.get('root') != root or marker.get('purpose') != 'TNSM container startup':
        raise ValueError('Runtime ownership marker mismatch')
    if marker.get('uid') != path.stat().st_uid or int(marker.get('uid', 0)) <= 0:
        raise ValueError('Runtime root is not owned by its declared ordinary SSH user')
    if os.geteuid() != 0 or int(os.environ.get('SUDO_UID', '-1')) != marker['uid']:
        raise PermissionError('Use only this SSH user existing noninteractive sudo authorization')
    return path, marker


def validate_https_proxy(https_proxy):
    if https_proxy is None:
        return None
    if not re.fullmatch(r'http://127\.0\.0\.1:[1-9][0-9]{0,4}', https_proxy):
        raise ValueError('Only a board-loopback experiment proxy is permitted')
    if int(https_proxy.rsplit(':', 1)[1]) > 65535:
        raise ValueError('Invalid board-loopback proxy port')
    return https_proxy


def environment(root, https_proxy=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('DOCKER_')}
    env.update(PATH=root+'/docker:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
               PYTHONDONTWRITEBYTECODE='1')
    https_proxy = validate_https_proxy(https_proxy)
    if https_proxy:
        env.update(HTTP_PROXY=https_proxy, HTTPS_PROXY=https_proxy,
                   http_proxy=https_proxy, https_proxy=https_proxy,
                   NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    return env


def invoke(argv, root, timeout=30, monitor_temperature=False):
    begun = time.monotonic()
    child = None
    try:
        child = subprocess.Popen(argv, env=environment(root), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, start_new_session=True)
        guard_error = None
        while True:
            remaining = timeout-(time.monotonic()-begun)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                output, error = child.communicate(timeout=min(.5, remaining) if monitor_temperature else remaining)
                break
            except subprocess.TimeoutExpired:
                if not monitor_temperature:
                    raise
                try:
                    cpu_sensors()
                except Exception as exc:
                    guard_error = repr(exc)
                    child.terminate()
                    try:
                        output, error = child.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        output, error = child.communicate()
                    break
        return dict(argv=argv, returncode=child.returncode, stdout=output, stderr=error,
                    temperature_error=guard_error,
                    elapsed_s=time.monotonic()-begun, ended_utc=utc())
    except subprocess.TimeoutExpired as exc:
        if child is not None:
            child.kill()
            output, error = child.communicate()
        else:
            output, error = '', ''
        return dict(argv=argv, returncode=None, stdout=output, stderr=error,
                    timeout_s=timeout, elapsed_s=time.monotonic()-begun, ended_utc=utc())


def record(root, name, value):
    # Names are internal constants; every evidence file is create-only.
    path = Path(root)/name
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def cpu_sensors():
    sensors = []
    for name in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        kind = read(name+'/type') or ''
        lower = kind.lower()
        if ('cpu' in lower or 'soc' in lower) and 'pmic' not in lower:
            value = float(read(name+'/temp'))/1000.
            if not math.isfinite(value) or not -20 <= value < 65:
                raise RuntimeError('Identified CPU sensor unsafe or invalid: '+kind)
            sensors.append(dict(zone=os.path.basename(name), real_path=os.path.realpath(name),
                                kind=kind, cpu_c=value))
    if not sensors:
        raise RuntimeError('No identified readable CPU/SoC temperature sensor')
    return sensors


def process_identity(pid):
    try:
        raw = Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\x00')
        procstat = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, argv=[v.decode(errors='replace') for v in raw if v],
                    executable=os.readlink('/proc/%d/exe' % pid), state=procstat[0],
                    start_ticks=procstat[19])
    except (OSError, IndexError):
        return None


def preflight(root):
    path, owner = root_path(root)
    existing = [p for p in ['/var/run/docker.sock', '/run/docker.sock', '/var/run/docker.pid']
                if os.path.lexists(p)]
    dockerd_pids = [int(Path(p).parent.name) for p in glob.glob('/proc/[0-9]*/comm')
                    if read(p) == 'dockerd']
    if existing or dockerd_pids:
        raise RuntimeError('Existing Docker daemon/state: refuse to alter or share it')
    apparmor = read('/sys/module/apparmor/parameters/enabled')
    profiles = read('/sys/kernel/security/apparmor/profiles') or ''
    if apparmor == 'Y' and 'docker-default (enforce)' not in profiles:
        raise RuntimeError('Default Docker AppArmor profile is not preloaded; no host-security policy changes')
    if shutil.disk_usage(root).free < 2*1024**3:
        raise RuntimeError('Less than 2 GiB free for private retained runtime stores')
    arch = platform.machine()
    if arch not in ('aarch64', 'armv7l'):
        raise RuntimeError('Unexpected physical board architecture')
    if not Path('/sys/fs/cgroup').is_dir():
        raise RuntimeError('No cgroup filesystem')
    result = dict(utc=utc(), root=root, owner=owner, architecture=arch, kernel=platform.release(),
                  python=sys.version, os_release=read('/etc/os-release'), sensors=cpu_sensors(),
                  memory=read('/proc/meminfo'), disk_free=shutil.disk_usage(root).free,
                  apparmor_enabled=apparmor, cgroups=read('/proc/cgroups'),
                  ip_forward=read('/proc/sys/net/ipv4/ip_forward'),
                  ipv6_forward=read('/proc/sys/net/ipv6/conf/all/forwarding'))
    record(root, 'preflight.json', result)
    return result


def safe_archive_members(archive):
    members = archive.getmembers()
    files = []
    for member in members:
        if member.isdir() and member.name.rstrip('/') == 'docker':
            continue
        if not member.isfile() or not re.fullmatch(r'docker/[A-Za-z0-9_-]+', member.name):
            raise ValueError('Unexpected executable archive member: '+member.name)
        if member.size <= 0 or member.size > 180*1024**2:
            raise ValueError('Invalid executable archive member size')
        files.append(member)
    if len({m.name for m in files}) != len(files) or not {'docker/docker', 'docker/dockerd',
            'docker/containerd', 'docker/runc'}.issubset({m.name for m in files}):
        raise ValueError('Duplicate or incomplete Docker static executable set')
    return files


def bootstrap(root, https_proxy=None):
    root_path(root)
    initial = json.loads((Path(root)/'preflight.json').read_text())
    arch = 'aarch64' if initial['architecture'] == 'aarch64' else 'armhf'
    url = 'https://download.docker.com/linux/static/stable/%s/docker-%s.tgz' % (arch, DOCKER_VERSION)
    archive = Path(root)/'docker-static.tgz'
    started = time.monotonic()
    https_proxy = validate_https_proxy(https_proxy)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'https': https_proxy} if https_proxy else {}))
    attempts, total, checksum = [], None, None
    for attempt in range(1, 4):
        candidate = Path(root)/('docker-static.attempt-%d.tgz' % attempt)
        attempt_started = time.monotonic()
        try:
            with opener.open(url, timeout=60) as response, candidate.open('xb') as output:
                if not response.geturl().startswith('https://download.docker.com/'):
                    raise ValueError('Unexpected download redirect')
                declared = response.headers.get('Content-Length')
                declared = int(declared) if declared is not None else None
                total = 0
                last_check = time.monotonic()
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 150*1024**2:
                        raise ValueError('Unexpected Docker archive size')
                    output.write(chunk)
                    if time.monotonic()-last_check >= .5:
                        cpu_sensors()
                        last_check = time.monotonic()
            if declared is not None and total != declared:
                raise EOFError('Static runtime byte count differs from Content-Length')
            checksum = hashlib.sha256(candidate.read_bytes()).hexdigest()
            # Force the complete gzip stream to be decoded before accepting it.
            # This detects a truncated transfer before any executable is created.
            expanded = 0
            with tarfile.open(str(candidate)) as source:
                members = safe_archive_members(source)
                for member in members:
                    with source.extractfile(member) as stream:
                        for chunk in iter(lambda: stream.read(1024*1024), b''):
                            expanded += len(chunk)
            attempts.append(dict(attempt=attempt, complete=True, bytes=total,
                content_length=declared, expanded_bytes=expanded, sha256=checksum,
                elapsed_s=time.monotonic()-attempt_started))
            candidate.replace(archive)
            break
        except Exception as exc:
            attempts.append(dict(attempt=attempt, complete=False,
                bytes=candidate.stat().st_size if candidate.exists() else 0,
                error=repr(exc), elapsed_s=time.monotonic()-attempt_started))
            if candidate.exists():
                candidate.unlink()
            # A temperature/sensor failure is a safety stop, not a retriable
            # transport fault.
            if 'temperature' in str(exc).lower() or 'sensor' in str(exc).lower():
                record(root, 'bootstrap_download_attempts.json', attempts)
                raise
            if attempt < 3:
                time.sleep(2*attempt)
    record(root, 'bootstrap_download_attempts.json', attempts)
    if not archive.exists() or total is None or checksum is None:
        raise RuntimeError('Static runtime download failed after three validated attempts')
    (Path(root)/'docker').mkdir()
    with tarfile.open(str(archive)) as source:
        for member in safe_archive_members(source):
            dest = Path(root)/member.name
            with source.extractfile(member) as src, dest.open('xb') as out:
                last_check = time.monotonic()
                while True:
                    chunk = src.read(1024*1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    if time.monotonic()-last_check >= .5:
                        cpu_sensors()
                        last_check = time.monotonic()
            dest.chmod(0o755)
    result = dict(utc=utc(), url=url, bytes=total, sha256=checksum,
                  elapsed_s=time.monotonic()-started,
                  version=invoke([root+'/docker/dockerd', '--version'], root))
    record(root, 'bootstrap.json', result)
    if result['version']['returncode']:
        raise RuntimeError('Downloaded runtime executable cannot run on this board')
    return result


def start(root, store, https_proxy=None):
    root_path(root)
    cpu_sensors()
    p = runtime_paths(root, store)
    base = Path(p['base'])
    base.mkdir()
    Path(p['client']).mkdir()
    Path(p['config']).write_text(json.dumps(daemon_config(root, store), indent=2))
    cmd = [root+'/docker/dockerd', '--config-file', p['config']]
    check = invoke(cmd+['--validate'], root)
    record(root, store+'/config_validation.json', check)
    if check['returncode']:
        raise RuntimeError('Private daemon configuration is not supported')
    with Path(p['log']).open('xb') as logfile:
        child = subprocess.Popen(cmd, env=environment(root, https_proxy), stdin=subprocess.DEVNULL,
            stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    identity = process_identity(child.pid)
    record(root, store+'/process.json', dict(identity=identity, utc=utc()))
    deadline = time.monotonic()+60
    last = None
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('Private daemon exited; retained dockerd.log')
        last = invoke(docker_prefix(root, store)+['info', '--format', '{{json .}}'], root, 5)
        if last['returncode'] == 0:
            info = json.loads(last['stdout'])
            if info.get('DockerRootDir') != p['data'] or info.get('Driver') != 'vfs':
                raise RuntimeError('Runtime storage boundary mismatch')
            record(root, store+'/runtime_info.json', last)
            return last
        time.sleep(.5)
    record(root, store+'/start_failure.json', last)
    raise RuntimeError('Private runtime did not become ready')


def pull(root, store, reference):
    root_path(root)
    if reference != 'docker.io/library/python:3.11.16-slim-bookworm':
        image_digest(reference)
    cpu_sensors()
    prefix = docker_prefix(root, store)
    before = invoke(prefix+['image', 'ls', '--all', '--quiet'], root)
    if before['returncode'] or before['stdout'].strip():
        raise RuntimeError('Image-uncached measurement requires a confirmed empty private store')
    started = utc()
    result = invoke(prefix+['pull', reference], root, 900, monitor_temperature=True)
    inspection = invoke(prefix+['image', 'inspect', reference], root)
    record_data = dict(started_utc=started, reference=reference, before=before,
        download=result, inspection=inspection,
        boundary='Private-store image-uncached; registry, Internet and OS page cache uncontrolled')
    try:
        record_data['temperature_after'] = cpu_sensors()
    except Exception as exc:
        record_data['temperature_error'] = repr(exc)
    # Preserve acquisition failures, never silently retry/replace a timed observation.
    record(root, store+'/acquisition.json', record_data)
    if record_data.get('temperature_error') or result.get('temperature_error'):
        raise RuntimeError('Acquisition temperature guard; raw download observation retained')
    if result['returncode'] != 0 or inspection['returncode'] != 0:
        raise RuntimeError('Image acquisition failed; raw observation retained')
    data = json.loads(inspection['stdout'])[0]
    digests = [v for v in data.get('RepoDigests', []) if 'python@sha256:' in v]
    if len(digests) != 1:
        raise RuntimeError('Image digest is absent or ambiguous')
    record_data['resolved_digest'] = image_digest(digests[0])
    record(root, store+'/resolved_image.json', record_data)
    return record_data


def stop(root, store):
    root_path(root)
    p = runtime_paths(root, store)
    prior = json.loads((Path(p['base'])/'process.json').read_text())['identity']
    if not prior or prior['executable'] != root+'/docker/dockerd':
        raise RuntimeError('Daemon process ownership was not established')
    current = process_identity(prior['pid'])
    if current and (current['start_ticks'] != prior['start_ticks'] or current['executable'] != prior['executable']):
        raise RuntimeError('Daemon PID identity changed; do not signal it')
    listing = invoke(docker_prefix(root, store)+['ps', '--all', '--quiet'], root, 15) if current else None
    stopped_containers = []
    if listing and listing['returncode'] == 0:
        for identity in listing['stdout'].split():
            if not re.fullmatch(r'[0-9a-f]{12,64}', identity):
                raise RuntimeError('Unexpected container identity; no broad cleanup')
            inspected = invoke(docker_prefix(root, store)+['inspect', identity], root)
            if inspected['returncode']:
                raise RuntimeError('Cannot establish leftover container ownership')
            data = json.loads(inspected['stdout'])[0]
            label = data.get('Config', {}).get('Labels', {}).get('tnsm.container-study')
            if label != root.rsplit('/', 1)[1] or not re.fullmatch(r'/tnsm-p\d{2}-b\d{2}', data.get('Name', '')):
                raise RuntimeError('Container is not from this study; refuse to stop it')
            stopped = invoke(docker_prefix(root, store)+['stop', '--time', '10', identity], root, 30)
            stopped_containers.append(dict(identity=identity, inspection=inspected, stop=stopped))
            if stopped['returncode']:
                raise RuntimeError('Study container did not stop')
    if current:
        os.kill(prior['pid'], signal.SIGTERM)
        deadline = time.monotonic()+35
        while time.monotonic() < deadline:
            current = process_identity(prior['pid'])
            if not current or current['state'] == 'Z':
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Daemon did not stop; no forced broad termination')
    mounts = [line for line in (read('/proc/self/mountinfo') or '').splitlines() if p['base'] in line]
    survivors = []
    for item in glob.glob('/proc/[0-9]*/cmdline'):
        pid = int(Path(item).parent.name)
        if pid == os.getpid():
            continue
        identity = process_identity(pid)
        if identity and identity['state'] != 'Z' and p['base'] in ' '.join(identity['argv']):
            survivors.append(identity)
    before = json.loads((Path(root)/'preflight.json').read_text())
    unchanged = (before['ip_forward'] == read('/proc/sys/net/ipv4/ip_forward') and
                 before['ipv6_forward'] == read('/proc/sys/net/ipv6/conf/all/forwarding'))
    result = dict(utc=utc(), daemon_stopped=True, remaining_mounts=mounts,
                  remaining_store_processes=survivors, forwarding_unchanged=unchanged,
                  stopped_containers=stopped_containers, runtime_data_retained=p['base'])
    record(root, store+'/stop.json', result)
    if mounts or survivors or not unchanged:
        raise RuntimeError('Private runtime cleanup or network-state verification failed')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'bootstrap', 'start', 'pull', 'stop'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--store', default='pull_00')
    parser.add_argument('--reference')
    parser.add_argument('--https-proxy')
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            result = preflight(args.root)
        elif args.action == 'bootstrap':
            result = bootstrap(args.root, args.https_proxy)
        elif args.action == 'pull':
            result = pull(args.root, args.store, args.reference)
        elif args.action == 'start':
            result = start(args.root, args.store, args.https_proxy)
        else:
            result = globals()[args.action](args.root, args.store)
        print(json.dumps(dict(ok=True, result=result), allow_nan=False), flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=repr(exc), utc=utc())), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
