import numpy as np
from src.forecast_online import HORIZON, boundaries, past_scale, purged_masks, sample_matrix


def test_scaling_is_prefix_invariant():
    rng=np.random.default_rng(1)
    p=rng.uniform(1,10,(400,3)).astype(np.float32)
    q=p.copy();q[200:]*=1e6
    np.testing.assert_array_equal(past_scale(p)[:201],past_scale(q)[:201])


def test_four_minute_targets_do_not_cross_splits():
    t=np.arange(64,300)
    train,val=purged_masks(t,288)
    boundary,_=boundaries(288)
    assert (t[train]+HORIZON<=boundary).all()
    assert (t[val]>=boundary).all()
    assert (t[val]+HORIZON<=288).all()
    assert not (train&val).any()


def test_future_mutation_cannot_change_current_features():
    p=np.arange(1200,dtype=np.float32).reshape(400,3)
    q=p.copy();q[200:]+=1e8
    x,_,_=sample_matrix(p,np.arange(64,201),160)
    y,_,_=sample_matrix(q,np.arange(64,201),160)
    np.testing.assert_array_equal(x,y)
