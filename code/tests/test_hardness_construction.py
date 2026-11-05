"""Executable check of the two constructions; not a substitute for the proof."""
from itertools import product


def attainable(work):
    sums={0}
    for value in work: sums |= {s+value for s in tuple(sums)}
    return sums


def test_equal_and_strictly_heterogeneous_partition_constructions():
    for n in range(1,7):
        for work in product(range(1,5),repeat=n):
            total=sum(work)
            if total%2: continue  # These map to a fixed infeasible instance.
            deadline=total//2
            subset=attainable(work)
            partition=deadline in subset
            equal=any(s<=deadline and total-s<=deadline for s in subset)
            # Dummy job has work A, pinned to rate-2 node. Capacity is 2A.
            heterogeneous=any(s<=deadline and total-s+deadline<=2*deadline for s in subset)
            assert partition==equal==heterogeneous
