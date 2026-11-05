import numpy as np
from src.scheduler_core import batch_work, within_phase_release


def test_batch_work_is_additive_and_zero_empty():
    for work in (.001, .1, 1., 51.5, 9000.):
        for count in range(65):
            assert batch_work(work, count) == work*count
            assert np.isclose(batch_work(work, count), sum(batch_work(work, 1) for _ in range(count)))


def test_split_batches_never_escape_phase_or_reorder():
    rng = np.random.default_rng(604)
    for chunks in (1, 4, 13, 1000):
        for phase in range(128):
            release = np.array([within_phase_release(120., phase, 128, i, chunks, rng.random())
                                for i in range(chunks)])
            assert np.all(np.diff(release) > 0)
            assert release[0] >= 120.+60.*phase/128
            assert release[-1] < 120.+60.*(phase+1)/128

