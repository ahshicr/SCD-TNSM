import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from src.container_trial_design import (BLOCKS, PORTS, archive_gate, campaign_gate, daemon_config,
    design, docker_prefix, image_digest, runtime_paths, validate_root, worker_command)
from src.container_runtime_remote import bootstrap, safe_archive_members, validate_https_proxy
from src.run_container_trials import allowed_registry_host


ROOT = '/var/tmp/tnsm-container.abcdefgh'
DIGEST = 'python@sha256:'+'a'*64
SENSORS = [{'zone': 'thermal_zone0', 'real_path': '/sys/devices/virtual/thermal/thermal_zone0'}]


def test_complete_matched_design_has_all_four_generations_and_equal_inputs():
    rows = design()
    assert len(rows) == 160 and rows == design()
    for port in PORTS:
        for block in range(BLOCKS):
            pair = [r for r in rows if r['port'] == port and r['block'] == block]
            assert len(pair) == 2
            assert {r['method'] for r in pair} == {'host_process', 'cached_container'}
            assert pair[0]['first_input'] == pair[1]['first_input']
            assert pair[0]['resident_inputs'] == pair[1]['resident_inputs']
            assert len(pair[0]['resident_inputs']) == 20
            assert all(r['temperature_guard_c'] == 65 and r['cooldown_seconds'] == 10 for r in pair)


def test_registry_proxy_allowlist_is_suffix_bounded():
    assert allowed_registry_host('registry-1.docker.io')
    assert allowed_registry_host('auth.docker.io')
    assert allowed_registry_host('production.cloudflare.docker.com')
    assert allowed_registry_host('x.r2.cloudflarestorage.com')
    assert not allowed_registry_host('docker.io')
    assert not allowed_registry_host('docker.io.attacker.example')
    assert not allowed_registry_host('example.com')


def test_remote_helper_accepts_only_board_loopback_proxy():
    assert validate_https_proxy('http://127.0.0.1:12345') == 'http://127.0.0.1:12345'
    assert validate_https_proxy(None) is None
    for value in ['http://0.0.0.0:12345', 'https://127.0.0.1:12345',
                  'http://127.0.0.1:0', 'http://127.0.0.1:99999']:
        with pytest.raises(ValueError):
            validate_https_proxy(value)


@pytest.mark.parametrize('path', ['/var/tmp', '/', '/tmp/tnsm-container.abcdefgh',
    '/var/tmp/tnsm-container.abcdefgh/../other', ROOT+'/child', ROOT+';id'])
def test_runtime_root_rejects_broad_traversal_and_shell_paths(path):
    with pytest.raises(ValueError):
        validate_root(path)


def test_daemon_and_client_use_no_default_roots_or_network_management():
    config = daemon_config(ROOT, 'pull_00')
    assert config['bridge'] == 'none' and config['storage-driver'] == 'vfs'
    for key in ['iptables', 'ip6tables', 'ip-forward', 'ip-masq', 'userland-proxy']:
        assert config[key] is False
    assert not config['features']['containerd-snapshotter']
    paths = runtime_paths(ROOT, 'pull_00')
    assert config['hosts'] == ['unix://'+paths['socket']]
    assert all(config[k].startswith(ROOT+'/') for k in ['data-root', 'exec-root', 'pidfile'])
    assert paths['socket'] in docker_prefix(ROOT, 'pull_00')[2]
    with pytest.raises(ValueError):
        daemon_config(ROOT, '../docker')


def test_container_is_nonroot_without_network_caps_host_socket_or_writable_sensor():
    cmd = worker_command(ROOT, 'pull_04', DIGEST, 'tnsm-p05-b00', SENSORS, 1000, 1000)
    for flag in ['--read-only', '--network=none', '--cap-drop=ALL', '--security-opt=no-new-privileges']:
        assert flag in cmd
    assert '--privileged' not in cmd and '--pid=host' not in cmd
    mounts = [cmd[i+1] for i, value in enumerate(cmd) if value == '--mount']
    assert all(m.endswith(',readonly') for m in mounts)
    assert all('docker.sock' not in m for m in mounts)
    with pytest.raises(ValueError):
        worker_command(ROOT, 'pull_04', DIGEST, 'tnsm-p05-b00', SENSORS, 0, 0)
    with pytest.raises(ValueError):
        worker_command(ROOT, 'pull_04', DIGEST, 'tnsm-p05-b00', [], 1000, 1000)
    with pytest.raises(ValueError):
        worker_command(ROOT, 'pull_04', DIGEST, 'tnsm-p05-b00',
            [{'zone': 'thermal_zone0', 'real_path': '/sys/devices/../../etc/thermal_zone0'}], 1000, 1000)


def test_inference_refuses_mutable_or_non_python_image_references():
    for value in ['python:latest', 'python:3.11', 'ubuntu@sha256:'+'a'*64, DIGEST+'x']:
        with pytest.raises(ValueError):
            image_digest(value)
    assert image_digest('docker.io/library/'+DIGEST) == 'docker.io/library/'+DIGEST


def archive_bytes(extra=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name in ['docker/docker', 'docker/dockerd', 'docker/containerd', 'docker/runc']:
            item = tarfile.TarInfo(name)
            item.size = 1
            tf.addfile(item, io.BytesIO(b'x'))
        if extra:
            tf.addfile(extra)
    buf.seek(0)
    return buf


def test_static_archive_rejects_traversal_symlinks_devices_and_duplicate_members():
    with tarfile.open(fileobj=archive_bytes()) as tf:
        assert len(safe_archive_members(tf)) == 4
    bad = [tarfile.TarInfo('../etc/example'), tarfile.TarInfo('docker/docker')]
    link = tarfile.TarInfo('docker/link')
    link.type, link.linkname = tarfile.SYMTYPE, '/usr/bin/python3'
    bad.append(link)
    device = tarfile.TarInfo('docker/device')
    device.type = tarfile.CHRTYPE
    bad.append(device)
    for item in bad:
        with tarfile.open(fileobj=archive_bytes(item)) as tf, pytest.raises(ValueError):
            safe_archive_members(tf)


def test_bootstrap_retries_a_truncated_archive_before_creating_executables(tmp_path, monkeypatch):
    from src import container_runtime_remote as runtime
    root = str(tmp_path)
    (tmp_path/'preflight.json').write_text(json.dumps({'architecture': 'aarch64'}))
    good = io.BytesIO()
    with tarfile.open(fileobj=good, mode='w:gz') as tf:
        for name in ['docker/docker', 'docker/dockerd', 'docker/containerd', 'docker/runc']:
            item = tarfile.TarInfo(name)
            item.size = 1
            tf.addfile(item, io.BytesIO(b'x'))
    good_bytes = good.getvalue()

    class Response(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.headers = {'Content-Length': str(len(value))}
        def geturl(self):
            return 'https://download.docker.com/linux/static/stable/aarch64/example.tgz'

    values = iter([Response(good_bytes[:50]), Response(good_bytes)])
    class Opener:
        def open(self, url, timeout):
            return next(values)
    monkeypatch.setattr(runtime, 'root_path', lambda value: (tmp_path, {}))
    monkeypatch.setattr(runtime.urllib.request, 'build_opener', lambda *args: Opener())
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(runtime, 'invoke', lambda *args, **kwargs: {'returncode': 0})
    result = bootstrap(root)
    attempts = json.loads((tmp_path/'bootstrap_download_attempts.json').read_text())
    assert [item['complete'] for item in attempts] == [False, True]
    assert result['bytes'] == len(good_bytes)
    assert (tmp_path/'docker'/'dockerd').exists()


def write_gate_fixture(directory):
    status = dict(complete=True, completed=[dict(ordinal=0, complete=True)],
                  power_sampler=dict(errors=[], thread_stopped=True))
    (directory/'manifest.json').write_text(json.dumps({'planned_trials': 1}))
    (directory/'campaign_status.json').write_text(json.dumps(status))
    trial = directory/'trial_000'
    trial.mkdir()
    summary = dict(complete=True, error=None, cleanup={'5': dict(errors=[],
        stop_acknowledged=True, directory_removed=True)})
    (trial/'summary.json').write_text(json.dumps(summary))
    return status, summary


def test_campaign_gate_blocks_live_incomplete_sampler_and_worker_cleanup(tmp_path):
    with pytest.raises(RuntimeError, match='still active'):
        campaign_gate(tmp_path)
    status, summary = write_gate_fixture(tmp_path)
    assert campaign_gate(tmp_path)['complete']
    status['power_sampler']['thread_stopped'] = False
    (tmp_path/'campaign_status.json').write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match='sampler cleanup'):
        campaign_gate(tmp_path)
    status['power_sampler']['thread_stopped'] = True
    status['completed'][0]['ordinal'] = 2
    (tmp_path/'campaign_status.json').write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match='identities'):
        campaign_gate(tmp_path)
    status['completed'][0]['ordinal'] = 0
    (tmp_path/'campaign_status.json').write_text(json.dumps(status))
    summary['cleanup']['5']['stop_acknowledged'] = False
    (tmp_path/'trial_000'/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match='worker cleanup'):
        campaign_gate(tmp_path)


def test_archive_gate_requires_completed_statistics_and_matching_archive(tmp_path):
    campaign, archive = tmp_path/'campaign', tmp_path/'archive'
    campaign.mkdir()
    archive.mkdir()
    with pytest.raises(RuntimeError, match='not finished'):
        archive_gate(campaign, archive)
    (campaign/'analysis').mkdir()
    analysis = dict(complete=True, trials=280, blocks=10, parity_failures=0)
    (campaign/'analysis'/'audit.json').write_text(json.dumps(analysis))
    report = dict(campaign_complete=True, analysis_audit_present=True, source=str(campaign))
    (archive/'archive_verification.json').write_text(json.dumps(report))
    assert archive_gate(campaign, archive)['campaign_complete']
    analysis['trials'] = 279
    (campaign/'analysis'/'audit.json').write_text(json.dumps(analysis))
    with pytest.raises(RuntimeError, match='full-matrix'):
        archive_gate(campaign, archive)
    report['source'] = str(tmp_path/'unrelated')
    (archive/'archive_verification.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='different'):
        archive_gate(campaign, archive)


def test_runtime_subprocess_thermal_guard_preserves_failed_command_output(monkeypatch):
    from src import container_runtime_remote as runtime
    class Child:
        returncode = None
        stopped = False
        def communicate(self, timeout=None):
            if not self.stopped:
                raise runtime.subprocess.TimeoutExpired(['test-only-no-process'], timeout)
            return 'partial transfer', 'safely stopped'
        def terminate(self):
            self.stopped, self.returncode = True, -15
        def kill(self):
            self.stopped, self.returncode = True, -9
    child = Child()
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: child)
    def guarded():
        raise RuntimeError('test-only CPU >= 65 C')
    monkeypatch.setattr(runtime, 'cpu_sensors', guarded)
    record = runtime.invoke(['test-only-no-process'], ROOT, monitor_temperature=True)
    assert record['returncode'] == -15 and record['stdout'] == 'partial transfer'
    assert '65 C' in record['temperature_error']


def test_startup_statistics_do_not_count_resident_requests_as_independent_blocks():
    from src.summarize_container_trials import interval
    assert interval([2.]*20) == dict(mean=2., halfwidth95=0., n=20)
    assert interval([-2.]*20)['mean'] == -2.
    with pytest.raises(ValueError):
        interval([2.]*400)
    with pytest.raises(ValueError):
        interval([2.]*19+[float('nan')])


def test_measured_startup_profiles_preserve_class_mapping_and_bound_central(tmp_path):
    from src.startup_profiles import derive_startup_profiles
    boards = []
    for port, mean, halfwidth, surcharge in [
            (5, 100., 10., 20.), (9, 200., 20., 40.),
            (7, 300., 30., 30.), (8, 400., 40., 50.)]:
        boards.append(dict(port=port, metrics=dict(ready_ms=dict(
            cached_container=dict(mean=mean, halfwidth95=halfwidth, n=20),
            paired_container_minus_host=dict(mean=surcharge, halfwidth95=1., n=20)))))
    container = tmp_path/'container.json'
    container.write_text(json.dumps(boards), encoding='utf-8')
    (tmp_path/'audit.json').write_text(json.dumps(dict(
        complete=True, activation_count=len(PORTS)*2*BLOCKS)), encoding='utf-8')
    activation = tmp_path/'activation.json'
    activation.write_text(json.dumps(dict(activation=[
        dict(port=12, worker_activation_s=dict(mean=.25, ci95_half=.001, n=20)),
        dict(port=20, worker_activation_s=dict(mean=.30, ci95_half=.001, n=20)),
    ])), encoding='utf-8')
    derived = derive_startup_profiles(container, activation)
    classes = ('pi5', 'rdk_x5', 'pi4', 'pi3', 'pi2', 'jetson_nano')
    low = [derived['profiles']['measured_low'][name]['mean_s'] for name in classes]
    central = [derived['profiles']['measured_central'][name]['mean_s'] for name in classes]
    high = [derived['profiles']['measured_high'][name]['mean_s'] for name in classes]
    assert central == pytest.approx([.1, .285, .2, .3, .4, .335])
    assert low == pytest.approx([.09, .269, .18, .27, .36, .319])
    assert high == pytest.approx([.11, .301, .22, .33, .44, .351])
    assert all(a <= b <= c for a, b, c in zip(low, central, high))
    assert all(
        derived['profiles'][profile][name]['sd_s'] > 0
        for profile in ('measured_low', 'measured_central', 'measured_high')
        for name in classes
    )


def test_measured_startup_profile_maps_every_strict_node_and_recovers_moments(tmp_path):
    from src.startup_profiles import deployment_with_startup_profile
    from src.strict_config import load_deployment
    root = Path(__file__).resolve().parents[1]
    deployment = load_deployment(root/'config'/'strict_deployment_20260914.json')
    profile = {
        name: {'mean_s': 1.0+index, 'sd_s': 0.1*(index+1), 'basis': 'test'}
        for index, name in enumerate(('pi5', 'pi4', 'pi3', 'pi2', 'rdk_x5', 'jetson_nano'))
    }
    evidence = {'profiles': {
        'measured_low': profile,
        'measured_central': profile,
        'measured_high': profile,
    }}
    adjusted = deployment_with_startup_profile(deployment, evidence, 'measured_central', 'a'*64)
    assert adjusted.source_sha256 == 'a'*64
    for index, device_class in enumerate(adjusted.device_class):
        mean = np.exp(adjusted.startup_mu_log_s[index] + .5*adjusted.startup_sigma_log_s[index]**2)
        variance = (np.exp(adjusted.startup_sigma_log_s[index]**2)-1)*np.exp(
            2*adjusted.startup_mu_log_s[index]+adjusted.startup_sigma_log_s[index]**2)
        assert mean == pytest.approx(profile[str(device_class)]['mean_s'])
        assert np.sqrt(variance) == pytest.approx(profile[str(device_class)]['sd_s'])


def test_activation_audit_rejects_missing_requests_bad_parity_and_bad_timing():
    from copy import deepcopy
    from src.summarize_container_trials import audit_activation
    spec = design()[0]
    requests = [dict(id=i, input_index=index, status='ok', computed=1,
        max_relative_error=1e-7, before={'cpu_c': 45.}, after={'cpu_c': 46.},
        controller_send_start=1.+i, controller_send_end=1.01+i, controller_received=1.05+i,
        controller_latency_s=.05, compute_start=50.+i, compute_end=50.02+i)
        for i,index in enumerate([spec['first_input']]+spec['resident_inputs'])]
    observation = dict(spec=spec, complete=True, cleanup=dict(stop_acknowledged=True,
        errors=[], exit_status=0), requests=requests, launch_to_ready_s=.1,
        launch_to_first_prediction_s=.2, model_send_s=.05)
    result = audit_activation(observation, spec)
    assert result['ready_ms'] == 100 and result['resident_response_ms'] == pytest.approx(50)
    bad = deepcopy(observation)
    bad['requests'].pop()
    with pytest.raises(ValueError, match='Missing'):
        audit_activation(bad, spec)
    bad = deepcopy(observation)
    bad['requests'][0]['max_relative_error'] = 1.
    with pytest.raises(ValueError, match='parity'):
        audit_activation(bad, spec)
    bad = deepcopy(observation)
    bad['requests'][0]['controller_received'] = -1.
    with pytest.raises(ValueError, match='timing'):
        audit_activation(bad, spec)
