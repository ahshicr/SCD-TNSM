"""Read-only inventory of experiment capabilities, using previously pinned hosts.

No package installation, sudo configuration, reboot, or benchmark load is applied.
The private key is supplied via an environment variable and is never persisted.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

REMOTE = r'''
import glob,json,os,platform,shutil,subprocess,sys
def read(p):
    try:
        with open(p) as f: return f.read().replace(chr(0),'').strip()
    except (OSError,IOError): return None
def run(a):
    try:
        p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=5)
        return {'status':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except Exception as e: return {'error':str(e)}
thermal=[]
for p in glob.glob('/sys/class/thermal/thermal_zone*'):
    thermal.append({'path':p,'type':read(p+'/type'),'temp_millic':read(p+'/temp'),
                    'trips':{os.path.basename(t):read(t) for t in glob.glob(p+'/trip_point_*')}})
hwmon=[]
for p in glob.glob('/sys/class/hwmon/hwmon*'):
    vals={}
    for pattern in ('power*_input','curr*_input','in*_input','*_label','temp*_input','fan*_input'):
        vals.update({os.path.basename(f):read(f) for f in glob.glob(p+'/'+pattern)})
    hwmon.append({'path':p,'name':read(p+'/name'),'values':vals})
iio=[]
for p in glob.glob('/sys/bus/i2c/drivers/ina3221x/*/iio:device*')+glob.glob('/sys/bus/iio/devices/iio:device*'):
    vals={os.path.basename(f):read(f) for pattern in ('in_power*_input','in_current*_input','in_voltage*_input','rail_name*','in_*_label') for f in glob.glob(p+'/'+pattern)}
    if vals: iio.append({'path':p,'values':vals})
freq=[]
for p in glob.glob('/sys/devices/system/cpu/cpufreq/policy*'):
    freq.append({k:read(p+'/'+k) for k in ('affected_cpus','scaling_driver','scaling_governor','scaling_min_freq','scaling_max_freq','scaling_cur_freq')})
modules=run([sys.executable,'-c',"import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n) is not None for n in ['numpy','torch','onnxruntime','tensorflow','tflite_runtime','cv2']}))"])
print(json.dumps({'model':read('/proc/device-tree/model'),'python':sys.version,'os':read('/etc/os-release'),
 'kernel':platform.release(),'arch':platform.machine(),'memory':read('/proc/meminfo').splitlines()[:3],
 'load':read('/proc/loadavg'),'thermal':thermal,'hwmon':hwmon,'iio':iio,'cpufreq':freq,
 'tools':{k:shutil.which(k) for k in ('vcgencmd','tegrastats','docker','podman','iperf3','tc','gcc','timeout')},
 'modules':modules,'sudo_noninteractive':run(['sudo','-n','true']),
 'vcgencmd_throttled':run(['vcgencmd','get_throttled']) if shutil.which('vcgencmd') else None,
 'vcgencmd_pmic':run(['vcgencmd','pmic_read_adc']) if shutil.which('vcgencmd') else None,
 'link':{os.path.basename(p):{'speed':read(p+'/speed'),'state':read(p+'/operstate')} for p in glob.glob('/sys/class/net/*')}
}))
'''


def connect(row, key):
    class Pin(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, hostkey):
            actual = 'SHA256:' + base64.b64encode(hashlib.sha256(hostkey.asbytes()).digest()).decode().rstrip('=')
            if actual != row['host_key_sha256']:
                raise paramiko.SSHException('Pinned host identity changed; not authenticating')
            client.get_host_keys().add(hostname, hostkey.get_name(), hostkey)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(Pin())
    client.connect(row['ip'], username=row['username'], key_filename=key,
                   timeout=5, banner_timeout=6, auth_timeout=8,
                   allow_agent=False, look_for_keys=False)
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    key = os.environ['EDGE_SSH_KEY']
    rows = json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    rows = [r for r in rows if r['port'] not in (2, 15) and r.get('ssh_login') is True]
    def probe(row):
        result = {k: row[k] for k in ('port', 'ip', 'username', 'host_key_sha256')}
        client = None
        try:
            client = connect(row, key)
            _, out, err = client.exec_command('python3 -c ' + shlex.quote(REMOTE), timeout=45)
            data = out.read().decode('utf-8')
            status = out.channel.recv_exit_status()
            if status: raise RuntimeError(err.read().decode('utf-8', 'replace'))
            result['capabilities'] = json.loads(data)
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            if client: client.close()
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(probe, rows))
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'read_only': True,
              'excluded': {'2': 'wireless router confirmed by author', '15': 'uplink'}, 'nodes': result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in result:
        c = row.get('capabilities', {})
        print(json.dumps({'port': row['port'], 'model': c.get('model'),
                         'thermal_zones': len(c.get('thermal', [])), 'hwmon': c.get('hwmon'),
                         'iio': c.get('iio'), 'sudo': c.get('sudo_noninteractive'),
                         'modules': c.get('modules'), 'error': row.get('error')}), flush=True)


if __name__ == '__main__':
    main()
