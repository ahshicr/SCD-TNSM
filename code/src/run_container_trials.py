"""Full paired container/process experiment, gated on completed policy trials.

Only an explicit invocation performs board operations. No password is saved.
Runtime images and executables are left in a named, stopped private directory
until their locally copied evidence and safe cleanup have been checked.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import socket
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from src.board_capabilities import connect
from src.board_inference_client import BoardClient
from src.container_trial_design import (BLOCKS, IMAGE_TAG, PORTS, PULLS, SEED,
    archive_gate, campaign_gate, design, validate_root, worker_command)

SOURCE_NAMES = ['run_container_trials.py', 'container_trial_design.py',
    'container_runtime_remote.py', 'summarize_container_trials.py', 'board_capabilities.py', 'board_inference_client.py',
    'board_inference_worker.py', 'portable_inference.py']

REGISTRY_PROXY_SUFFIXES = ('.docker.io', '.docker.com',
    '.cloudflarestorage.com', '.cloudfront.net')


def allowed_registry_host(host):
    host = host.rstrip('.').lower()
    return any(host.endswith(suffix) and host != suffix[1:]
               for suffix in REGISTRY_PROXY_SUFFIXES)


class LoopbackRegistryProxy:
    """Expose an existing controller-loopback HTTPS proxy over pinned SSH.

    The remote listener is board-loopback only. CONNECT requests are limited
    to Docker authentication, registry and content-delivery domains on 443.
    """
    def __init__(self, transport):
        configured = urllib.request.getproxies().get('https')
        parsed = urllib.parse.urlparse(configured or '')
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or parsed.port is None or parsed.username or parsed.password):
            raise RuntimeError('A credential-free controller-loopback HTTPS proxy is required')
        self.transport = transport
        self.upstream = (parsed.hostname, parsed.port)
        self.lock = threading.Lock()
        self.requests = []
        self.errors = []
        self.threads = []
        self.remote_port = transport.request_port_forward('127.0.0.1', 0, self._handler)

    def _handler(self, channel, origin, server):
        worker = threading.Thread(target=self._serve, args=(channel,), daemon=True)
        self.threads.append(worker)
        worker.start()

    def _serve(self, channel):
        upstream = None
        host = None
        sent = received = 0
        try:
            channel.settimeout(30)
            header = bytearray()
            while b'\r\n\r\n' not in header:
                chunk = channel.recv(4096)
                if not chunk:
                    raise ConnectionError('Proxy client closed before CONNECT headers')
                header.extend(chunk)
                if len(header) > 16384:
                    raise ValueError('Oversized proxy request headers')
            first = bytes(header).split(b'\r\n', 1)[0].decode('ascii', 'strict')
            match = re.fullmatch(r'CONNECT ([A-Za-z0-9.-]+):([0-9]+) HTTP/1\.[01]', first)
            if not match or int(match.group(2)) != 443 or not allowed_registry_host(match.group(1)):
                channel.sendall(b'HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n')
                raise ValueError('CONNECT destination is outside the Docker registry allowlist')
            host = match.group(1).lower()
            upstream = socket.create_connection(self.upstream, timeout=15)
            upstream.sendall(header)
            channel.settimeout(None)
            upstream.settimeout(None)
            while True:
                readable, _, _ = select.select([channel, upstream], [], [], 30)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    if source is channel:
                        upstream.sendall(data)
                        sent += len(data)
                    else:
                        channel.sendall(data)
                        received += len(data)
        except Exception as exc:
            with self.lock:
                self.errors.append(type(exc).__name__+': '+str(exc))
        finally:
            if upstream is not None:
                upstream.close()
            channel.close()
            with self.lock:
                self.requests.append(dict(host=host, sent_bytes=sent, received_bytes=received))

    def close(self):
        self.transport.cancel_port_forward('127.0.0.1', self.remote_port)
        for worker in self.threads:
            worker.join(timeout=5)
        return dict(remote_bind='127.0.0.1', remote_port=self.remote_port,
                    allowlist=list(REGISTRY_PROXY_SUFFIXES), requests=self.requests,
                    errors=self.errors,
                    boundary='HTTPS CONNECT bytes over the pinned SSH session; payload not inspected or logged')


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def command(ssh, argv, timeout=90):
    began = time.monotonic()
    channel = ssh.get_transport().open_session(timeout=10)
    channel.exec_command(' '.join(shlex.quote(str(a)) for a in argv))
    output, error = bytearray(), bytearray()
    try:
        deadline = began+timeout
        while True:
            while channel.recv_ready():
                output.extend(channel.recv(65536))
            while channel.recv_stderr_ready():
                error.extend(channel.recv_stderr(65536))
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Remote action exceeded %s s; inspect retained runtime before retrying' % timeout)
            time.sleep(.02)
        return dict(argv=argv, returncode=channel.recv_exit_status(),
            stdout=output.decode(errors='replace'), stderr=error.decode(errors='replace'),
            elapsed_s=time.monotonic()-began, utc=utc())
    finally:
        channel.close()


class ActivationClient(BoardClient):
    """Reuse byte-identical worker framing, but not its provisioning/SSH lifetime."""
    def __init__(self, ssh, argv, model):
        self.channel = ssh.get_transport().open_session(timeout=10)
        self.channel.settimeout(90)
        self.cleanup = {'stop_acknowledged': False, 'errors': []}
        self.started = time.monotonic()
        self.channel.exec_command(' '.join(shlex.quote(str(a)) for a in argv))
        self.executed = time.monotonic()
        self.send(dict(type='init', model=model, temperature_limit_c=65.))
        self.init_sent = time.monotonic()
        self.ready = self.receive()
        self.ready_received = time.monotonic()
        if self.ready.get('type') != 'ready':
            raise RuntimeError('Deployment worker did not initialize')

    def close(self):
        try:
            self.send(dict(type='stop'))
            self.channel.settimeout(20)
            self.cleanup['stop_acknowledged'] = self.receive().get('type') == 'stopped'
            deadline = time.monotonic()+25
            while not self.channel.exit_status_ready() and time.monotonic() < deadline:
                time.sleep(.05)
            if not self.channel.exit_status_ready():
                raise RuntimeError('Worker channel did not exit after stop acknowledgement')
            self.cleanup['exit_status'] = self.channel.recv_exit_status()
            if self.cleanup['exit_status'] != 0:
                raise RuntimeError('Worker/CLI exited unsuccessfully')
        except Exception as exc:
            self.cleanup['errors'].append(repr(exc))
        finally:
            self.channel.close()
        return self.cleanup


def assert_temperature(telemetry):
    value = telemetry.get('cpu_c')
    if value is None or not np.isfinite(value) or not -20 <= value < 65:
        raise RuntimeError('CPU temperature safety latch or missing identified sensor')


def activation(ssh, argv, spec, model, features, expected):
    client = None
    result = dict(spec=spec, started_utc=utc(), argv=argv, complete=False, requests=[])
    try:
        # Allocate before initializing so a failed startup channel is also closed.
        client = ActivationClient.__new__(ActivationClient)
        client.__init__(ssh, argv, model)
        result.update(worker_ready=client.ready,
            launch_to_ready_s=client.ready_received-client.started,
            exec_request_s=client.executed-client.started,
            model_send_s=client.init_sent-client.executed,
            model_sent_to_ready_s=client.ready_received-client.init_sent)
        assert_temperature(client.ready['telemetry'])
        indices = [spec['first_input']]+spec['resident_inputs']
        for identity, index in enumerate(indices):
            request = client.infer(identity, features[[index]])
            request['input_index'] = index
            predictions = np.asarray(request.get('predictions', []))
            parity = (float(np.max(np.abs(predictions-expected[index])/max(abs(expected[index]), 1.)))
                      if len(predictions) == 1 else None)
            request['max_relative_error'] = parity
            result['requests'].append(request)
            if identity == 0:
                result['launch_to_first_prediction_s'] = request['controller_received']-client.started
            if request.get('status') != 'ok' or request.get('computed') != 1:
                raise RuntimeError('Worker guarded or failed: '+str(request.get('status')))
            if parity is None or not np.isfinite(parity) or parity >= 1e-4:
                raise RuntimeError('Actual trained-model numerical parity failed')
            assert_temperature(request['before'])
            assert_temperature(request['after'])
        result['complete'] = True
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        if client is not None and hasattr(client, 'channel'):
            if hasattr(client, 'ready') and client.ready.get('type') == 'ready':
                result['cleanup'] = client.close()
            else:
                client.channel.close()
                result['cleanup'] = {'stop_acknowledged': False, 'errors': ['Activation failed before readiness']}
        result['ended_utc'] = utc()
    if result.get('cleanup', {}).get('errors') or not result.get('cleanup', {}).get('stop_acknowledged'):
        result['complete'] = False
    return result


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def collect_evidence(ssh, root, local):
    """Copy only named protocol/runtime log files, never image stores or secrets."""
    local.mkdir(exist_ok=False)
    copied, failures = [], []
    with ssh.open_sftp() as ftp:
        for relative in ['', *['pull_%02d' % n for n in range(PULLS)]]:
            directory = root+('/'+relative if relative else '')
            try:
                entries = ftp.listdir_attr(directory)
            except OSError:
                continue
            for item in entries:
                if not item.filename.endswith(('.json', '.log')):
                    continue
                remote = directory+'/'+item.filename
                target = local/(relative+'_'+item.filename if relative else item.filename)
                if item.st_size > 20*1024**2:
                    failures.append({'path': remote, 'error': 'Unexpectedly large evidence file'})
                    continue
                try:
                    ftp.get(remote, str(target))
                    copied.append(dict(remote=remote, local=target.name, bytes=target.stat().st_size,
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
                except OSError as exc:
                    failures.append(dict(path=remote, error=repr(exc)))
    dump(local/'collection.json', dict(copied=copied, failures=failures,
         boundary='Named runtime evidence only; generated executables and private image stores retained remotely'))
    if failures:
        raise RuntimeError('Runtime evidence collection incomplete')


def run(args):
    # Gate before creating output, opening SSH, sudo or downloading anything.
    sources = source_hashes()
    if args.wait:
        print(json.dumps(dict(event='waiting_for_policy_campaign_statistics_and_archive',
            campaign=str(args.after_campaign), archive=str(args.after_archive),
            board_operations_started=False)), flush=True)
        while not (args.after_campaign/'campaign_status.json').exists():
            time.sleep(30)
    campaign_gate(args.after_campaign)
    if args.wait:
        deadline = time.monotonic()+3600
        while not (args.after_archive/'archive_verification.json').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Policy archive did not finish within one hour; no board operations started')
            time.sleep(30)
    archive_gate(args.after_campaign, args.after_archive)
    if source_hashes() != sources:
        raise RuntimeError('Container code changed while waiting; restart explicitly with reviewed source')
    if str(args.out).startswith('\\\\'):
        raise ValueError('Collect experimental timing logs on local disk')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'source_snapshot').mkdir()
    for name in SOURCE_NAMES:
        (args.out/'source_snapshot'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    rows = json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    rows = {r['port']: r for r in rows if r.get('ssh_login')}
    if any(port not in rows for port in PORTS):
        raise ValueError('All four predeclared board identities must be in the pinned inventory')
    model_bytes = (args.model/'model.json').read_bytes()
    model = json.loads(model_bytes)
    examples = np.load(args.model/'validation_inputs.npz')
    features, expected = examples['features'], examples['expected']
    specs = design()
    manifest = dict(started_utc=utc(), seed=SEED, trials=specs, planned_activations=len(specs),
        pulls_per_board=PULLS, source_sha256=sources, image_tag=IMAGE_TAG,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256((args.model/'validation_inputs.npz').read_bytes()).hexdigest(),
        hypothesis='Deployment-stack activation contrast, not pure isolation overhead; Python versions recorded',
        statistics='20 paired blocks per board; not 400 independent resident-startup repetitions',
        cache_boundary='Fresh private image stores; no Internet, CDN or OS cache flushing',
        after_policy_campaign=str(args.after_campaign.resolve()),
        after_verified_archive=str(args.after_archive.resolve()))
    dump(args.out/'manifest.json', manifest)
    key = os.environ['EDGE_SSH_KEY']
    boards = []
    try:
        for port in PORTS:
            ssh, root, active_store, registry_proxy = None, None, None, None
            local = args.out/('port_%02d' % port)
            local.mkdir()
            board = dict(port=port, complete=False, activations=0, cleanup=[])
            boards.append(board)
            action_count = 0
            try:
                if source_hashes() != sources:
                    raise RuntimeError('Container experiment source hash changed')
                ssh = connect(rows[port], key)
                registry_proxy = LoopbackRegistryProxy(ssh.get_transport())
                board['registry_proxy'] = dict(remote_bind='127.0.0.1',
                    remote_port=registry_proxy.remote_port,
                    boundary='Docker-only HTTPS CONNECT over the pinned SSH session')
                sudo = command(ssh, ['sudo', '-n', 'true'])
                dump(local/'sudo_preflight.json', sudo)
                if sudo['returncode']:
                    raise RuntimeError('Existing noninteractive sudo is unavailable')
                created = command(ssh, ['mktemp', '-d', '/var/tmp/tnsm-container.XXXXXXXX'])
                if created['returncode']:
                    raise RuntimeError('Cannot create dedicated temporary runtime root')
                root = validate_root(created['stdout'].strip())
                board['remote_runtime_root'] = root
                uid = command(ssh, ['id', '-u'])
                gid = command(ssh, ['id', '-g'])
                owner = dict(root=root, purpose='TNSM container startup', uid=int(uid['stdout']), gid=int(gid['stdout']))
                with ssh.open_sftp() as ftp:
                    ftp.mkdir(root+'/worker')
                    for name in ['container_trial_design.py', 'container_runtime_remote.py']:
                        ftp.put(str(Path(__file__).with_name(name)), root+'/'+name)
                    for name in ['board_inference_worker.py', 'portable_inference.py']:
                        ftp.put(str(Path(__file__).with_name(name)), root+'/worker/'+name)
                    with ftp.file(root+'/owner.json', 'wx') as handle:
                        handle.write(json.dumps(owner))
                def remote(action, store='pull_00', reference=None, allow_acquisition_failure=False):
                    nonlocal action_count
                    argv = ['sudo', '-n', 'python3', '-B', root+'/container_runtime_remote.py', action,
                            '--root', root, '--store', store]
                    if reference is not None:
                        argv += ['--reference', reference]
                    if action in ('bootstrap', 'start'):
                        argv += ['--https-proxy', 'http://127.0.0.1:%d' % registry_proxy.remote_port]
                    answer = command(ssh, argv, 1200 if action in ('pull', 'bootstrap') else 120)
                    dump(local/('action_%03d_%s.json' % (action_count, action)), answer)
                    action_count += 1
                    if answer['returncode'] and allow_acquisition_failure:
                        return dict(failed=True, error=answer['stdout'], raw_action=answer)
                    if answer['returncode']:
                        raise RuntimeError('Private runtime '+action+' failed: '+answer['stdout'][-1500:])
                    return json.loads(answer['stdout'])['result']
                preflight = remote('preflight')
                board['preflight'] = preflight
                remote('bootstrap')
                digest = None
                for pull_id in range(PULLS):
                    active_store = 'pull_%02d' % pull_id
                    remote('start', active_store)
                    acquisition = remote('pull', active_store, digest or IMAGE_TAG, True)
                    if acquisition.get('failed'):
                        board.setdefault('failed_acquisitions', []).append(dict(pull_id=pull_id,
                            error=acquisition['error']))
                        board['cleanup'].append(remote('stop', active_store))
                        active_store = None
                        # A missing/unsafe sensor is a safety stop, not a network observation to skip.
                        if 'temperature' in acquisition['error'].lower() or 'sensor' in acquisition['error'].lower():
                            raise RuntimeError('Acquisition temperature safety stop')
                        continue
                    if digest is not None and acquisition['resolved_digest'] != digest:
                        raise RuntimeError('Immutable image identity changed between acquisitions')
                    digest = acquisition['resolved_digest']
                    board['image_digest'] = digest
                    if not board['activations']:
                        for spec in (x for x in specs if x['port'] == port):
                            if source_hashes() != sources:
                                raise RuntimeError('Container experiment source hash changed')
                            time.sleep(spec['cooldown_seconds'])
                            if spec['method'] == 'host_process':
                                argv = ['python3', '-B', '-u', root+'/worker/board_inference_worker.py']
                            else:
                                argv = ['sudo', '-n']+worker_command(root, active_store, digest,
                                    'tnsm-p%02d-b%02d' % (port, spec['block']), preflight['sensors'],
                                    owner['uid'], owner['gid'])
                            observation = activation(ssh, argv, spec, model, features, expected)
                            dump(local/('activation_b%02d_%s.json' % (spec['block'], spec['method'])), observation)
                            print(json.dumps(dict(event='container_activation', port=port, block=spec['block'],
                                method=spec['method'], complete=observation['complete'],
                                error=observation.get('error'))), flush=True)
                            if not observation['complete']:
                                raise RuntimeError('Incomplete activation; no favorable-case replacement')
                            board['activations'] += 1
                    board['cleanup'].append(remote('stop', active_store))
                    active_store = None
                board['complete'] = board['activations'] == 2*BLOCKS
                if not board['complete']:
                    raise RuntimeError('All five image acquisitions failed; no activation evidence available')
            except Exception as exc:
                board['error'] = repr(exc)
                raise
            finally:
                if ssh and root:
                    if active_store:
                        try:
                            board['cleanup'].append(remote('stop', active_store))
                        except Exception as exc:
                            board['cleanup_error'] = repr(exc)
                    try:
                        collect_evidence(ssh, root, local/'runtime_evidence')
                    except Exception as exc:
                        board['evidence_error'] = repr(exc)
                    board['runtime_files_retained'] = root
                if registry_proxy:
                    board['registry_proxy_audit'] = registry_proxy.close()
                if ssh:
                    ssh.close()
                dump(local/'summary.json', board)
            if board.get('cleanup_error') or board.get('evidence_error'):
                raise RuntimeError('Runtime cleanup/evidence unresolved; do not start another board')
    finally:
        dump(args.out/'campaign_status.json', dict(ended_utc=utc(), boards=boards,
            complete=len(boards) == len(PORTS) and all(b['complete'] and not b.get('error') and
                not b.get('cleanup_error') and not b.get('evidence_error') for b in boards)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--after-campaign', type=Path, required=True)
    parser.add_argument('--after-archive', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--model', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    run(parser.parse_args())
