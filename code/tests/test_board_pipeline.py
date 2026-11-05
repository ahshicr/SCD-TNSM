import queue
import threading
import pytest
from src.board_pipeline import BoardPipeline


class FakeClient:
    def __init__(self):
        self.writes = []
        self.messages = queue.Queue()
        self.release = threading.Event()

    def send(self, metadata, raw):
        self.writes.append((metadata, raw))
        self.messages.put(dict(id=metadata['id'], type='result', status='ok'))

    def receive(self):
        self.release.wait(3)
        return self.messages.get(timeout=3)

    def close(self):
        return dict(stop_acknowledged=True)


def test_full_duplex_bounded_ordered_and_clean_shutdown():
    client = FakeClient()
    pipe = BoardPipeline(client=client)
    futures = [pipe.submit(i, [[float(i)]*66], 1024) for i in range(4)]
    with pytest.raises(queue.Full):
        pipe.submit(9, [[0.]*66])
    client.release.set()
    results = [f.result(timeout=5) for f in futures]
    assert [r['id'] for r in results] == list(range(4))
    assert all(len(raw) == 1024 for _, raw in client.writes)
    assert all(r['controller_enqueued'] <= r['controller_send_start'] <=
               r['controller_send_end'] <= r['controller_received'] for r in results)
    assert pipe.inflight == 0
    assert pipe.submit(4, [[0.]*66]).result(timeout=5)['id'] == 4
    closed = pipe.close()
    assert closed['pipeline_threads_stopped']
    assert closed['pipeline_failure'] is None


def test_invalid_depth_duplicate_and_oversized_payload():
    with pytest.raises(ValueError):
        BoardPipeline(client=FakeClient(), max_inflight=5)
    client = FakeClient()
    pipe = BoardPipeline(client=client)
    future = pipe.submit('a', [[0.]*66])
    with pytest.raises(ValueError):
        pipe.submit('a', [[0.]*66])
    with pytest.raises(ValueError):
        pipe.submit('b', [[0.]*66], 17*1024*1024)
    client.release.set()
    future.result(timeout=5)
    assert pipe.close()['pipeline_threads_stopped']
