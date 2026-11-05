"""Bounded full-duplex SSH pipeline for the unchanged persistent model worker."""
from __future__ import annotations
from concurrent.futures import Future
import queue
import struct
import threading
import time
from src.board_inference_client import BoardClient


class BoardPipeline:
    def __init__(self, row=None, key=None, model=None, max_inflight=4, client=None):
        if not 1 <= max_inflight <= 4:
            raise ValueError('In-flight bound must fit the audited remote queue')
        self.client = client if client is not None else BoardClient(row, key, model)
        self.limit = max_inflight
        self.pending = {}
        self.condition = threading.Condition()
        self.transmit = queue.Queue()
        self.closing = False
        self.failure = None
        self.sender = threading.Thread(target=self._sender, name='tnsm-payload-sender', daemon=True)
        self.receiver = threading.Thread(target=self._receiver, name='tnsm-result-receiver', daemon=True)
        self.sender.start()
        self.receiver.start()

    @property
    def inflight(self):
        with self.condition:
            return len(self.pending)

    def submit(self, identity, features, payload_bytes=0, service_id='mlp'):
        batch = len(features)
        if not 1 <= batch <= 64:
            raise ValueError('Invalid model batch')
        raw = struct.pack('<%df' % (66*batch), *(float(v) for row in features for v in row))
        size = max(len(raw), payload_bytes)
        if size > 16*1024*1024:
            raise ValueError('Payload exceeds audited worker bound')
        raw += bytes(size-len(raw))
        metadata = dict(type='infer', id=identity, batch=batch, payload_bytes=size, service_id=service_id)
        future = Future()
        entry = dict(future=future, controller_enqueued=time.monotonic(),
                     sent=threading.Event())
        with self.condition:
            if self.closing or self.failure:
                raise RuntimeError('Pipeline is closed or failed')
            if identity in self.pending:
                raise ValueError('Duplicate in-flight identity')
            if len(self.pending) >= self.limit:
                raise queue.Full('In-flight bound reached')
            self.pending[identity] = entry
            self.transmit.put((metadata, raw))
            self.condition.notify_all()
        return future

    def _fail(self, exc):
        with self.condition:
            self.failure = repr(exc)
            for entry in self.pending.values():
                if not entry['future'].done():
                    entry['future'].set_exception(exc)
                entry['sent'].set()
            self.pending.clear()
            self.condition.notify_all()

    def _sender(self):
        try:
            while True:
                item = self.transmit.get()
                if item is None:
                    return
                metadata, raw = item
                with self.condition:
                    entry = self.pending.get(metadata['id'])
                    if entry is None:
                        return
                entry['controller_send_start'] = time.monotonic()
                self.client.send(metadata, raw)
                entry['controller_send_end'] = time.monotonic()
                entry['sent'].set()
        except Exception as exc:
            self._fail(exc)

    def _receiver(self):
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.closing and not self.failure:
                        self.condition.wait()
                    if not self.pending and (self.closing or self.failure):
                        return
                result = self.client.receive()
                received = time.monotonic()
                with self.condition:
                    entry = self.pending.get(result.get('id'))
                    if entry is None:
                        raise ValueError('Unmatched result identity')
                if not entry['sent'].wait(60):
                    raise TimeoutError('Result observed without completed payload write')
                result.update(controller_received=received,
                    controller_send_start=entry['controller_send_start'],
                    controller_send_end=entry['controller_send_end'],
                    controller_enqueued=entry['controller_enqueued'],
                    controller_latency_s=received-entry['controller_send_start'])
                with self.condition:
                    del self.pending[result['id']]
                    entry['future'].set_result(result)
                    self.condition.notify_all()
        except Exception as exc:
            self._fail(exc)

    def close(self, timeout=120):
        deadline = time.monotonic()+timeout
        with self.condition:
            self.closing = True
            while self.pending and not self.failure:
                left = deadline-time.monotonic()
                if left <= 0:
                    self._fail(TimeoutError('Pipeline drain exceeded declared timeout'))
                    break
                self.condition.wait(min(left, 1))
            self.condition.notify_all()
        self.transmit.put(None)
        self.sender.join(timeout=5)
        self.receiver.join(timeout=5)
        if self.sender.is_alive() or self.receiver.is_alive():
            # Interrupt only this worker's channel, never another remote process.
            self.client.channel.close()
            self.sender.join(timeout=5)
            self.receiver.join(timeout=5)
        cleanup = self.client.close()
        cleanup['pipeline_failure'] = self.failure
        cleanup['pipeline_threads_stopped'] = not (self.sender.is_alive() or self.receiver.is_alive())
        return cleanup
