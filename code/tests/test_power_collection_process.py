import json
import sys
import time
from src.power_collection_process import ProcessPowerSampler


def test_isolated_collector_preserves_protocol_and_transmits_no_command_line_secret(tmp_path):
    worker=tmp_path/'worker.py'
    worker.write_text('''import json,sys
first=json.loads(sys.stdin.readline())
assert first['password']=='unit-test-placeholder'
print(json.dumps({'event':'ready'}),flush=True)
for line in sys.stdin:
 command=json.loads(line)
 if command['command']=='renew':
  print(json.dumps({'event':'renewed','sequence':command['sequence']}),flush=True)
  print(json.dumps({'event':'state','errors':[{'error':'observed support gap'}]}),flush=True)
 elif command['command']=='close':
  print(json.dumps({'event':'closed','result':{'samples':17,'errors':[{'error':'observed support gap'}],'thread_stopped':True}}),flush=True)
  break
''')
    directory=tmp_path/'records'
    directory.mkdir()
    sampler=ProcessPowerSampler(directory,'unit-test-placeholder',worker_command=[sys.executable,'-u',str(worker)])
    assert 'unit-test-placeholder' not in ' '.join(sampler.process.args)
    assert sampler.errors==[]
    sampler.renew_between_trials()
    deadline=time.monotonic()+3
    while not sampler.errors and time.monotonic()<deadline:
        time.sleep(.01)
    assert sampler.errors==[{'error':'observed support gap'}]
    result=sampler.close()
    assert result['samples']==17 and result['errors']==sampler.errors
    assert sampler.process.returncode==0
    assert 'unit-test-placeholder' not in (directory/'collector_stderr.txt').read_text()
