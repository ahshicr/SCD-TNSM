"""Prequential forecasting with past-only fitting, scales and purged labels.

Every deployment block is a complete day. The first two days use persistence;
later checkpoints are fit only on already observed history. The last 20% of
that history supplies validation. Four-minute labels never cross either split.
The entire 14-day trace remains in the scheduling experiment.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from src.forecast_v2 import HISTORY, HORIZON, build_pressure, make_model, regression_metrics


def boundaries(deployment_start: int) -> tuple[int, int]:
    validation_start = HISTORY + int((deployment_start - HISTORY) * 0.8)
    return validation_start, deployment_start


def past_scale(pressure: np.ndarray) -> np.ndarray:
    """At t, includes [0,t) only. One fixed unit at t=0 is explicit bootstrap."""
    result = np.ones(pressure.shape, dtype=np.float64)
    result[1:] = np.cumsum(pressure[:-1], axis=0, dtype=np.float64) / np.arange(1, len(pressure))[:, None]
    return np.maximum(result, 1e-3).astype(np.float32)


def purged_masks(times: np.ndarray, deployment_start: int):
    val_start, stop = boundaries(deployment_start)
    return ((times + HORIZON <= val_start),
            ((times >= val_start) & (times + HORIZON <= stop)))


def sample_matrix(pressure: np.ndarray, times: np.ndarray, static_end: int):
    classes = pressure.shape[1]
    log = np.log1p(pressure)
    cls_ids = np.tile(np.arange(classes), len(times))
    ts = np.repeat(times, classes)
    history = log[ts[:, None] - HISTORY + np.arange(HISTORY)[None, :], cls_ids[:, None]]
    static = np.maximum(pressure[:static_end].mean(axis=0), 1e-3)
    x = np.column_stack((history, np.log1p(static[cls_ids]), cls_ids / max(1, classes - 1))).astype(np.float32)
    return x, ts, cls_ids


def run(aggregate: Path, out: Path, epochs: int, seed: int):
    out.mkdir(parents=True, exist_ok=False)
    pressure = build_pressure(aggregate)
    minutes, classes = pressure.shape
    scale = past_scale(pressure)
    persistence = np.zeros_like(pressure)
    persistence[1:] = pressure[:-1]
    ewma = np.zeros_like(pressure)
    for t in range(1, minutes):
        values = pressure[max(0, t-16):t]
        weights = .82 ** np.arange(len(values)-1, -1, -1, dtype=np.float64)
        ewma[t] = np.average(values, axis=0, weights=weights)
    target = np.zeros_like(pressure)
    target_valid = np.arange(minutes) + HORIZON <= minutes
    for t in np.flatnonzero(target_valid): target[t] = pressure[t:t+HORIZON].mean(axis=0)
    prediction = persistence.copy()
    origin = np.full(minutes, -1, dtype=np.int32)
    folds = []
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.monotonic()
    for block in range(2880, minutes, 1440):
        stop = min(block + 1440, minutes)
        val_start, _ = boundaries(block)
        # No feature or target construction includes the deployment block.
        train_times = np.arange(HISTORY, block-HORIZON+1)
        x, ts, cls = sample_matrix(pressure, train_times, val_start)
        y = np.log1p(target[ts, cls] * HORIZON).astype(np.float32).reshape(-1, 1)
        tm, vm = purged_masks(ts, block)
        xm, xs = x[tm].mean(0), x[tm].std(0) + 1e-6
        ym, ys = y[tm].mean(0), y[tm].std(0) + 1e-6
        xn = torch.from_numpy((x-xm)/xs).to(device)
        yn = torch.from_numpy((y-ym)/ys).to(device)
        torch.manual_seed(seed+block)
        model = make_model(x.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss()
        ti = np.flatnonzero(tm)
        vi = np.flatnonzero(vm)
        rng = np.random.default_rng(seed+block)
        best = math.inf
        best_state = None
        history = []
        stale = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0.
            order = rng.permutation(ti)
            for begin in range(0, len(order), 4096):
                ids = order[begin:begin+4096]
                loss = loss_fn(model(xn[ids]), yn[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                train_loss += float(loss.detach()) * len(ids)
            model.eval()
            with torch.no_grad():
                validation = sum(float(loss_fn(model(xn[ids]), yn[ids]))*len(ids)
                                 for ids in (vi[i:i+8192] for i in range(0,len(vi),8192))) / len(vi)
            history.append({'epoch':epoch+1,'train_loss':train_loss/len(ti),'validation_loss':validation})
            if validation < best-1e-6:
                best = validation
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale = 0
            else: stale += 1
            if stale >= 8: break
        model.load_state_dict(best_state)
        test_x, test_t, test_cls = sample_matrix(pressure, np.arange(block,stop), val_start)
        with torch.no_grad():
            pred = np.concatenate([model(torch.from_numpy((test_x[i:i+8192]-xm)/xs).to(device)).cpu().numpy()
                                   for i in range(0,len(test_x),8192)])
        prediction[test_t,test_cls] = np.maximum(np.expm1(pred.reshape(-1)*ys.item()+ym.item()),0)/HORIZON
        origin[block:stop] = block
        checkpoint = {'model':best_state,'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys,
                      'static_scale':pressure[:val_start].mean(0),'fit_before_minute':block,
                      'train_labels_end_exclusive':val_start,'history':HISTORY,'horizon':HORIZON}
        torch.save(checkpoint,out/f'checkpoint_before_{block:05d}.pt')
        row = {'deployment_start':block,'deployment_stop':stop,'train_labels_end_exclusive':val_start,
               'validation_labels_end_exclusive':block,'train_samples':int(tm.sum()),
               'validation_samples':int(vm.sum()),'epochs':len(history),'training':history}
        folds.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='training'}),flush=True)
    learned = (origin>=0)&target_valid
    metrics = {name:regression_metrics(target[learned]*HORIZON,values[learned]*HORIZON)
               for name,values in [('model',prediction),('persistence',persistence),('ewma',ewma)]}
    np.savez_compressed(out/'forecast_prequential.npz',prediction=prediction,
        persistence_prediction=persistence,ewma_prediction=ewma,target=target,
        normalization_scale=scale,model_origin=origin,target_valid=target_valid)
    report = {'protocol':'daily expanding-window prequential; initial two days persistence',
              'minutes':minutes,'classes':classes,'seed':seed,'device':str(device),
              'metrics_on_unseen_deployment_blocks':metrics,'folds':folds,
              'elapsed_seconds':time.monotonic()-start_time}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--aggregate',type=Path,default=Path('results/revision/workload_aggregate_16x4.npz'))
    p.add_argument('--out',type=Path,default=Path('results/revision/forecast_prequential_20260905'))
    p.add_argument('--epochs',type=int,default=48)
    p.add_argument('--seed',type=int,default=20260905)
    args=p.parse_args()
    run(args.aggregate,args.out,args.epochs,args.seed)
