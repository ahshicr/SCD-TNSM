from __future__ import annotations

"""Leakage-safe multi-class short-horizon demand forecasting.

The recovered implementation concatenated samples class by class and then made
one global 82/18 split.  That accidentally placed entire classes in the
validation tail.  This revision uses a chronological split inside every class,
selects the best validation checkpoint, evaluates against persistence and EWMA,
and exports class-specific forecasts plus calibrated residual uncertainty.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


HISTORY = 64
HORIZON = 4


def build_pressure(aggregate_path: Path) -> np.ndarray:
    data = np.load(aggregate_path)
    count = data["count"].sum(axis=2)
    compute = data["compute"].sum(axis=2)
    pressure = compute / np.maximum(count, 1.0)
    pressure *= count / 1000.0
    return pressure.astype(np.float32)


def build_samples(pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minutes, classes = pressure.shape
    rows: list[np.ndarray] = []
    targets: list[float] = []
    time_ids: list[int] = []
    class_ids: list[int] = []
    class_scale = np.maximum(pressure.mean(axis=0), 1e-3)
    for cls in range(classes):
        series = np.log1p(pressure[:, cls])
        static = np.asarray(
            [math.log1p(float(class_scale[cls])), cls / max(classes - 1, 1)],
            dtype=np.float32,
        )
        for t in range(HISTORY, minutes - HORIZON):
            rows.append(np.concatenate([series[t - HISTORY : t], static]))
            targets.append(np.log1p(float(pressure[t : t + HORIZON, cls].sum())))
            time_ids.append(t)
            class_ids.append(cls)
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(time_ids, dtype=np.int32),
        np.asarray(class_ids, dtype=np.int16),
    )


def masks(time_ids: np.ndarray, minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable_start = HISTORY
    usable_end = minutes - HORIZON
    train_end = usable_start + int(0.70 * (usable_end - usable_start))
    val_end = usable_start + int(0.85 * (usable_end - usable_start))
    return time_ids < train_end, (time_ids >= train_end) & (time_ids < val_end), time_ids >= val_end


def make_model(input_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.08),
        torch.nn.Linear(256, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = y.astype(np.float64).reshape(-1)
    pred = pred.astype(np.float64).reshape(-1)
    err = pred - y
    denom = np.maximum(np.abs(y) + np.abs(pred), 1e-6)
    ss_tot = float(np.square(y - y.mean()).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "smape_percent": float((200.0 * np.abs(err) / denom).mean()),
        "r2": float(1.0 - np.square(err).sum() / max(ss_tot, 1e-12)),
    }


def train_and_predict(
    aggregate_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    pressure = build_pressure(aggregate_path)
    x, y, time_ids, class_ids = build_samples(pressure)
    train_mask, val_mask, test_mask = masks(time_ids, pressure.shape[0])
    x_mean = x[train_mask].mean(axis=0, keepdims=True)
    x_std = x[train_mask].std(axis=0, keepdims=True) + 1e-6
    y_mean = y[train_mask].mean(axis=0, keepdims=True)
    y_std = y[train_mask].std(axis=0, keepdims=True) + 1e-6
    xn = ((x - x_mean) / x_std).astype(np.float32)
    yn = ((y - y_mean) / y_std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rng = np.random.default_rng(seed)

    x_tensor = torch.from_numpy(xn)
    y_tensor = torch.from_numpy(yn)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx].to(device, non_blocking=True)
            yb = y_tensor[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size * 2):
                idx = val_idx[start : start + batch_size * 2]
                xb = x_tensor[idx].to(device, non_blocking=True)
                yb = y_tensor[idx].to(device, non_blocking=True)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best_val - 1e-6:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    pred_n = np.zeros_like(yn)
    with torch.no_grad():
        for start in range(0, len(xn), batch_size * 2):
            xb = x_tensor[start : start + batch_size * 2].to(device, non_blocking=True)
            pred_n[start : start + len(xb)] = model(xb).cpu().numpy()
    pred_log = pred_n * y_std + y_mean
    pred = np.maximum(np.expm1(pred_log), 0.0).reshape(-1)
    truth = np.maximum(np.expm1(y), 0.0).reshape(-1)

    persistence = np.maximum(4.0 * pressure[time_ids - 1, class_ids], 0.0)
    ewma = np.empty_like(persistence)
    for idx, (t, cls) in enumerate(zip(time_ids, class_ids)):
        values = pressure[max(0, t - 16) : t, cls]
        weights = np.power(0.82, np.arange(len(values) - 1, -1, -1, dtype=np.float32))
        ewma[idx] = 4.0 * float(np.dot(values, weights) / max(weights.sum(), 1e-6))

    metrics = {
        "model": regression_metrics(truth[test_mask], pred[test_mask]),
        "persistence": regression_metrics(truth[test_mask], persistence[test_mask]),
        "ewma": regression_metrics(truth[test_mask], ewma[test_mask]),
    }
    class_rmse = np.zeros(pressure.shape[1], dtype=np.float32)
    for cls in range(pressure.shape[1]):
        mask = val_mask & (class_ids == cls)
        class_rmse[cls] = float(np.sqrt(np.square(pred[mask] - truth[mask]).mean()))

    prediction = np.zeros_like(pressure, dtype=np.float32)
    target = np.zeros_like(pressure, dtype=np.float32)
    persistence_prediction = np.zeros_like(pressure, dtype=np.float32)
    ewma_prediction = np.zeros_like(pressure, dtype=np.float32)
    for value, actual, persist, smooth, t, cls in zip(
        pred, truth, persistence, ewma, time_ids, class_ids
    ):
        prediction[t, cls] = float(value / HORIZON)
        target[t, cls] = float(actual / HORIZON)
        persistence_prediction[t, cls] = float(persist / HORIZON)
        ewma_prediction[t, cls] = float(smooth / HORIZON)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "input_dim": int(x.shape[1]),
            "history": HISTORY,
            "horizon": HORIZON,
            "seed": seed,
        },
        out_dir / "forecaster_v2.pt",
    )
    np.savez_compressed(
        out_dir / "forecast_class_v2.npz",
        prediction=prediction,
        target=target,
        persistence_prediction=persistence_prediction,
        ewma_prediction=ewma_prediction,
        class_rmse=class_rmse,
    )
    (out_dir / "forecast_training_v2.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary: dict[str, object] = {
        "samples": int(len(x)),
        "features": int(x.shape[1]),
        "classes": int(pressure.shape[1]),
        "split": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "type": "chronological within every class",
        },
        "epochs_completed": int(len(history)),
        "best_validation_loss": float(best_val),
        "device": str(device),
        "metrics_on_held_out_test": metrics,
        "class_validation_rmse": [float(x) for x in class_rmse],
    }
    (out_dir / "forecast_summary_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate", default="results/revision/workload_aggregate_16x4.npz"
    )
    parser.add_argument("--out-dir", default="results/revision/forecast_v2")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    summary = train_and_predict(
        Path(args.aggregate), Path(args.out_dir), args.epochs, args.batch_size, args.seed
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
