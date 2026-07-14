import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, accuracy_score

from ml_model.timefm.config import (
    MODELS_DIR,
    HEAD_PATH,
    FORECAST_HORIZON,
    MAX_EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    N_SPEED_CLASSES,
)
from ml_model.timefm.data import load_data
from ml_model.timefm.encoder import forecaster, OEEHead, SpeedHead

logger = logging.getLogger(__name__)


def compute_class_weights(y):
    classes = np.arange(N_SPEED_CLASSES)
    counts = np.array([(y == c).sum() for c in classes])
    counts = np.maximum(counts, 1)
    n = len(y)
    weights = n / (N_SPEED_CLASSES * counts.astype(float))
    weights = np.clip(weights, 0.3, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def precompute_forecasts(ctx_oee):
    logger.info(f"Pre-computing {len(ctx_oee)} forecasts...")
    forecasts = forecaster.forecast_batch(ctx_oee, horizon=FORECAST_HORIZON)
    logger.info(f"  done: shape={forecasts.shape}")
    return forecasts.astype(np.float32)


def train_heads(train_forecasts, train_cov_last, train_oee, train_speed,
                val_forecasts, val_cov_last, val_oee, val_speed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training heads on {device}")

    cov_mean = train_cov_last.mean(axis=0)
    cov_std = train_cov_last.std(axis=0) + 1e-8
    train_cov = (train_cov_last - cov_mean) / cov_std
    val_cov = (val_cov_last - cov_mean) / cov_std

    t_f = torch.tensor(train_forecasts)
    t_c = torch.tensor(train_cov)
    t_o = torch.tensor(train_oee)
    t_s = torch.tensor(train_speed, dtype=torch.long)
    v_f = torch.tensor(val_forecasts)
    v_c = torch.tensor(val_cov)
    v_o = torch.tensor(val_oee)
    v_s = torch.tensor(val_speed, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(t_f, t_c, t_o, t_s), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(v_f, v_c, v_o, v_s), batch_size=BATCH_SIZE)

    oee_head = OEEHead().to(device)
    speed_head = SpeedHead().to(device)
    class_weights = compute_class_weights(train_speed).to(device)

    opt = torch.optim.AdamW(
        list(oee_head.parameters()) + list(speed_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        oee_head.train()
        speed_head.train()
        train_loss = 0.0
        for f, c, o, s in train_loader:
            f, c, o, s = f.to(device), c.to(device), o.to(device), s.to(device)
            pred_oee = oee_head(f, c)
            pred_speed = speed_head(f, c)
            loss = mse_loss(pred_oee, o) + 0.3 * ce_loss(pred_speed, s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        oee_head.eval()
        speed_head.eval()
        val_loss = 0.0
        all_oee_true, all_oee_pred = [], []
        all_speed_true, all_speed_pred = []
        with torch.no_grad():
            for f, c, o, s in val_loader:
                f, c, o, s = f.to(device), c.to(device), o.to(device), s.to(device)
                pred_oee = oee_head(f, c)
                pred_speed = speed_head(f, c)
                loss = mse_loss(pred_oee, o) + 0.3 * ce_loss(pred_speed, s)
                val_loss += loss.item()
                all_oee_true.extend(o.cpu().numpy())
                all_oee_pred.extend(pred_oee.cpu().numpy())
                all_speed_true.extend(s.cpu().numpy())
                all_speed_pred.extend(pred_speed.argmax(1).cpu().numpy())

        r2 = r2_score(all_oee_true, all_oee_pred)
        acc = accuracy_score(all_speed_true, all_speed_pred)
        logger.info(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} | R²={r2:.4f} acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "oee_head": oee_head.state_dict(),
                "speed_head": speed_head.state_dict(),
                "cov_mean": cov_mean,
                "cov_std": cov_std,
                "val_r2": r2,
                "val_acc": acc,
            }
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, HEAD_PATH)
    logger.info(f"Saved heads: R²={best_state['val_r2']:.4f}, acc={best_state['val_acc']:.4f}")
    return best_state


def train(csv_path=None):
    forecaster.load()
    (Xo_tr, Xc_tr, yo_tr, ys_tr), (Xo_va, Xc_va, yo_va, ys_va) = load_data(csv_path)
    logger.info(f"Train: {len(Xo_tr)} windows, Val: {len(Xo_va)}")

    tr_f = precompute_forecasts(Xo_tr)
    va_f = precompute_forecasts(Xo_va)

    tr_clast = Xc_tr[:, -1, :]
    va_clast = Xc_va[:, -1, :]
    tr_oee_t10 = yo_tr[:, -1]
    va_oee_t10 = yo_va[:, -1]

    best = train_heads(tr_f, tr_clast, tr_oee_t10, ys_tr,
                       va_f, va_clast, va_oee_t10, ys_va)

    oee_head = OEEHead()
    speed_head = SpeedHead()
    oee_head.load_state_dict(best["oee_head"])
    speed_head.load_state_dict(best["speed_head"])
    return oee_head, speed_head, best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
