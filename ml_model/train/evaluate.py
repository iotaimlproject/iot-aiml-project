import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error

from ml_model.train.config import FEATURE_NAMES, TRAIN_CSV, TARGET_M1, TARGET_M2, SPEED_MAP

logger = logging.getLogger(__name__)


def load_val_data(csv_path=None, split_ratio=0.8):
    csv_path = csv_path or TRAIN_CSV
    df = pd.read_csv(csv_path)
    df = df.sort_values("Production_End_Time").reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    val_df = df.iloc[split_idx:].copy()
    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    y_m1 = val_df[TARGET_M1].values.astype(np.float32)
    y_m2 = np.array([SPEED_MAP[v] for v in val_df[TARGET_M2].values])
    return X_val, y_m1, y_m2


def scale_and_predict(X_val, model, scaler):
    X_s = scaler.transform(X_val)
    return model.predict(X_s, verbose=0)


def evaluate_m1(model, scaler, csv_path=None, split_ratio=0.8):
    X_val, y_m1, _ = load_val_data(csv_path, split_ratio)
    pred = scale_and_predict(X_val, model, scaler).ravel()
    r2 = r2_score(y_m1, pred)
    mae = mean_absolute_error(y_m1, pred)
    logger.info(f"M1 R²={r2:.4f}, MAE={mae:.4f}")
    return {"val_r2": round(float(r2), 4), "val_mae": round(float(mae), 4)}


def evaluate_m2(model, scaler, csv_path=None, split_ratio=0.8):
    X_val, _, y_m2 = load_val_data(csv_path, split_ratio)
    pred = np.argmax(scale_and_predict(X_val, model, scaler), axis=1)
    acc = accuracy_score(y_m2, pred)
    cm = confusion_matrix(y_m2, pred)
    logger.info(f"M2 accuracy: {acc*100:.1f}%")
    return {"val_accuracy": round(float(acc), 4), "confusion_matrix": cm.tolist()}
