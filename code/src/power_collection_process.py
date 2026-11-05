"""Separate-process PSE observation with the controller's monotonic clock.

Authentication travels through the child's input pipe, never a command line
or file. All samples, failed queries, and reconnects remain in the acquisition
directory. The same 2.5-second observed-support requirement remains active.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


class ProcessPowerSampler:
    def __init__(self,directory,password,worker_command=None):
        self._errors=[]
        self._ready=threading.Event()
        self._acks=queue.Queue()
        self._sequence=0
        self._closed=False
        command=worker_command or [sys.executable,'-u','-m','src.power_collection_process',
                                   '--directory',str(directory)]
        self._stderr=(Path(directory)/'collector_stderr.txt').open('x',encoding='utf-8')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self._reader=threading.Thread(target=self._read,name='power-process-observer',daemon=True)
        self._reader.start()
        self._send(dict(command='authenticate',password=password))
        if not self._ready.wait(30):
            self.process.terminate()
            self.process.wait(timeout=5)
            self._stderr.close()
            raise RuntimeError('Isolated PSE process did not become ready')
        if self._errors:
            raise RuntimeError('Isolated PSE process initialization failed')

    def _send(self,value):
        self.process.stdin.write(json.dumps(value)+'\n')
        self.process.stdin.flush()

    def _read(self):
        try:
            for line in self.process.stdout:
                value=json.loads(line)
                if value['event']=='ready':
                    self._ready.set()
                elif value['event']=='state':
                    self._errors=value['errors']
                elif value['event']=='error':
                    self._errors=[dict(error=value['error'])]
                    self._ready.set()
                else:
                    self._acks.put(value)
        except Exception as exc:
            self._errors=[dict(error='Collector protocol failure',detail=repr(exc))]
            self._ready.set()

    @property
    def errors(self):
        if not self._closed and self.process.poll() is not None and not self._errors:
            self._errors=[dict(error='Isolated acquisition process stopped unexpectedly')]
        return self._errors

    def renew_between_trials(self):
        if self.errors:
            raise RuntimeError('Cannot renew failed isolated PSE acquisition')
        self._sequence+=1
        self._send(dict(command='renew',sequence=self._sequence))
        result=self._acks.get(timeout=30)
        if result.get('event')!='renewed' or result.get('sequence')!=self._sequence:
            raise RuntimeError('Wrong PSE renewal acknowledgement')

    def close(self):
        if self._closed:
            raise RuntimeError('Isolated PSE process already closed')
        self._closed=True
        try:
            self._send(dict(command='close'))
            result=self._acks.get(timeout=30)
            if result.get('event')!='closed':
                raise RuntimeError('Missing isolated PSE shutdown record')
            self.process.wait(timeout=5)
            if self.process.returncode:
                raise RuntimeError('Isolated PSE process failed on shutdown')
            return result['result']
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            self._reader.join(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self._stderr.close()


def main():
    from src.redundant_power_sampler import RedundantPowerSampler
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    def emit(event,**value):
        print(json.dumps(dict(event=event,**value)),flush=True)
    initial=json.loads(sys.stdin.readline())
    if initial.get('command')!='authenticate':
        raise ValueError('Missing collector authentication message')
    sampler=None
    try:
        sampler=RedundantPowerSampler(args.directory,initial.pop('password'),recover=True)
        emit('ready')
        commands=queue.Queue()
        def incoming():
            for line in sys.stdin:
                commands.put(json.loads(line))
            commands.put(dict(command='close'))
        threading.Thread(target=incoming,daemon=True).start()
        while True:
            emit('state',errors=sampler.errors)
            try:
                command=commands.get(timeout=.25)
            except queue.Empty:
                continue
            if command['command']=='renew':
                sampler.renew_between_trials()
                emit('renewed',sequence=command['sequence'])
            elif command['command']=='close':
                result=sampler.close()
                sampler=None
                result['process_isolation']='Power collection uses a dedicated process on the same controller and its monotonic clock.'
                emit('closed',result=result)
                break
            else:
                raise ValueError('Unknown collector command')
    except BaseException as exc:
        emit('error',error=repr(exc))
        raise
    finally:
        initial.clear()
        if sampler is not None:
            sampler.close()


if __name__=='__main__':
    main()
