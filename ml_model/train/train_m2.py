import logging

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from ml_model.gpu_setup import configure_gpu
from ml_model.train.config import FEATURE_NAMES, M2_CONFIG, TRAIN_CSV, MODELS_DIR, TARGET_M2, SPEED_MAP
from ml_model.train.models import build_m2_classifier

logger = logging.getLogger(__name__)


def compute_class_weights(y):
    classes = np.arange(M2_CONFIG["n_classes"])
    counts = np.array([(y == c).sum() for c in classes])
    counts = np.maximum(counts, 1)
    n = len(y)
    weights = n / (M2_CONFIG["n_classes"] * counts.astype(float))
    weights = np.clip(weights, 0.3, 5.0)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def load_and_split(csv_path, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Production_End_Time").reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    y_train = np.array([SPEED_MAP[v] for v in train_df[TARGET_M2].values])
    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    y_val = np.array([SPEED_MAP[v] for v in val_df[TARGET_M2].values])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, y_train, X_val_s, y_val, scaler


def train(csv_path=None, split_ratio=0.8):
    configure_gpu()
    csv_path = csv_path or TRAIN_CSV
    logger.info(f"Loading data from {csv_path}")
    X_train, y_train, X_val, y_val, scaler = load_and_split(csv_path, split_ratio)
    dist = np.bincount(y_train)
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Distrib: {dist}")

    class_weights = compute_class_weights(y_train)
    logger.info(f"M2 class weights: {class_weights}")

    import tensorflow as tf
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_m2_classifier()
    es = EarlyStopping(
        monitor="val_loss",
        patience=M2_CONFIG["early_stopping_patience"],
        restore_best_weights=True,
    )
    h = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=M2_CONFIG["max_epochs"],
        batch_size=M2_CONFIG["batch_size"],
        class_weight=class_weights,
        callbacks=[es],
        verbose=0,
    )

    val_loss = min(h.history["val_loss"])
    val_acc = max(h.history["val_accuracy"])
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    final_acc = (y_pred == y_val).mean()
    logger.info(f"M2 done: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, final_acc={final_acc*100:.1f}%")

    model.save(str(MODELS_DIR / "m2_classifier.keras"))
    joblib.dump(scaler, str(MODELS_DIR / "scaler_m2.pkl"))
    logger.info(f"Saved m2_classifier.keras + scaler_m2.pkl to {MODELS_DIR}")

    return model, scaler, h.history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
