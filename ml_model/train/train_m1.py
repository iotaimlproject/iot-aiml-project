import logging

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from ml_model.train.config import FEATURE_NAMES, M1_CONFIG, TRAIN_CSV, MODELS_DIR, TARGET_M1
from ml_model.train.models import build_m1_regressor

logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]


def load_and_split(csv_path, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Production_End_Time").reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df[TARGET_M1].values.astype(np.float32)
    X_test = test_df[FEATURE_NAMES].values.astype(np.float32)
    y_test = test_df[TARGET_M1].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, y_train, X_test_s, y_test, scaler


def train(csv_path=None, split_ratio=0.8):
    csv_path = csv_path or TRAIN_CSV
    logger.info(f"Loading data from {csv_path}")
    X_train, y_train, X_test, y_test, scaler = load_and_split(csv_path, split_ratio)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(FEATURE_NAMES)}")
    logger.info(f"y range: [{y_train.min()}, {y_train.max()}]")

    import tensorflow as tf
    ensemble_preds = []

    for i, seed in enumerate(SEEDS):
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_m1_regressor()
        es = EarlyStopping(
            monitor="val_loss",
            patience=M1_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
        )
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=M1_CONFIG["max_epochs"],
            batch_size=M1_CONFIG["batch_size"],
            callbacks=[es],
            verbose=0,
        )
        preds = model.predict(X_test, verbose=0).ravel()
        ensemble_preds.append(preds)
        r2 = r2_score(y_test, preds)
        logger.info(f"  seed={seed}: test R²={r2:.4f}")

        model.save(str(MODELS_DIR / f"m1_oee_seed{seed}.keras"))
        logger.info(f"  saved m1_oee_seed{seed}.keras")

    y_pred = np.mean(ensemble_preds, axis=0)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"\n=== M1 Ensemble Results ===")
    logger.info(f"  R² test:  {r2:.4f}")
    logger.info(f"  MAE test: {mae:.2f} OEE pts")

    joblib.dump(scaler, str(MODELS_DIR / "scaler_m1.pkl"))
    logger.info(f"Saved scaler_m1.pkl to {MODELS_DIR}")
    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
