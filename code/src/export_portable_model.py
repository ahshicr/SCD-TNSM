"""Export one existing prequential checkpoint and verify real inference parity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.forecast_online import boundaries, sample_matrix
from src.forecast_v2 import build_pressure, make_model
from src.portable_inference import predict_one, validate_model


def run(checkpoint, aggregate, out):
    out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload['model']
    model = dict(layers=[dict(weight=state[f'{i}.weight'].tolist(), bias=state[f'{i}.bias'].tolist())
                        for i in (0, 3, 5, 7)],
                 x_mean=payload['x_mean'].reshape(-1).tolist(),
                 x_std=payload['x_std'].reshape(-1).tolist(),
                 y_mean=float(payload['y_mean'].item()), y_std=float(payload['y_std'].item()),
                 horizon=payload['horizon'], fit_before_minute=payload['fit_before_minute'])
    validate_model(model)
    pressure = build_pressure(aggregate)
    start = model['fit_before_minute']
    val_start, _ = boundaries(start)
    x, ts, classes = sample_matrix(pressure, np.arange(start, start+8), val_start)
    network = make_model(66).eval()
    network.load_state_dict(state)
    with torch.no_grad():
        yn = network(torch.from_numpy((x-payload['x_mean'])/payload['x_std'])).numpy().reshape(-1)
        expected = np.maximum(np.expm1(yn*model['y_std']+model['y_mean']), 0)/model['horizon']
    begun = time.monotonic()
    actual = np.array([predict_one(model, row) for row in x])
    relative = np.abs(actual-expected)/np.maximum(np.abs(expected), 1.)
    if relative.max() > 1e-4:
        raise ValueError(f'Portable inference parity failed: {relative.max()}')
    encoded = json.dumps(model, separators=(',', ':'), allow_nan=False).encode()
    (out/'model.json').write_bytes(encoded)
    np.savez_compressed(out/'validation_inputs.npz', features=x, expected=expected,
                        portable_prediction=actual, minute=ts, class_id=classes)
    with checkpoint.open('rb') as f:
        checkpoint_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  model_sha256=hashlib.sha256(encoded).hexdigest(), samples=len(x),
                  max_relative_error=float(relative.max()), mean_relative_error=float(relative.mean()),
                  max_absolute_error=float(np.abs(actual-expected).max()),
                  workstation_reference_elapsed_s=time.monotonic()-begun,
                  backend='Python standard library, CPU',
                  claim='Actual trained MLP; portable reference parity only, not board timing evidence')
    (out/'export_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('results/revision/forecast_prequential_20260905/checkpoint_before_02880.pt'))
    p.add_argument('--aggregate', type=Path, default=Path('results/revision/workload_aggregate_16x4.npz'))
    p.add_argument('--out', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    a = p.parse_args()
    run(a.checkpoint, a.aggregate, a.out)
