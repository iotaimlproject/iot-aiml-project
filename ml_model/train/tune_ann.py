import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from ml_model.train.config import FEATURE_NAMES, TRAIN_CSV, DIRECTION_MAP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_data(task="m1", split_ratio=0.8):
    df = pd.read_csv(TRAIN_CSV).sort_values(["Batch_Part_No", "Part_SLNo"]).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    if task == "m1":
        df["OEE_dir"] = np.sign(df["Delta_OEE"]).astype(int)
        train_df["OEE_dir"] = np.sign(train_df["Delta_OEE"]).astype(int)
        val_df["OEE_dir"] = np.sign(val_df["Delta_OEE"]).astype(int)
        y_train = (train_df["OEE_dir"].values + 1).astype(int)
        y_val = (val_df["OEE_dir"].values + 1).astype(int)
    else:
        y_train = np.array([DIRECTION_MAP[v] for v in train_df["Speed_Direction"].values])
        y_val = np.array([DIRECTION_MAP[v] for v in val_df["Speed_Direction"].values])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, y_train, X_val_s, y_val


def build_ann(arch, dropout_rate, lr, activation="relu", use_bn=False):
    model = Sequential(name="tune_ann")
    model.add(InputLayer(shape=(len(FEATURE_NAMES),)))
    for i, units in enumerate(arch):
        model.add(Dense(units, activation=activation))
        if use_bn:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def compute_weights(y):
    classes = np.arange(3)
    counts = np.array([(y == c).sum() for c in classes])
    counts = np.maximum(counts, 1)
    n = len(y)
    w = n / (3 * counts.astype(float))
    w = np.clip(w, 0.3, 5.0)
    return {int(c): float(v) for c, v in zip(classes, w)}


configs = [
    # (arch, dropout, lr, epochs, patience, use_bn, activation, label)
    ([128, 64, 32],         0.2,  0.001, 300, 25, False, "relu",  "128-64-32_d0.2_lr1e-3"),
    ([128, 64, 32],         0.3,  0.0005, 400, 30, False, "relu", "128-64-32_d0.3_lr5e-4"),
    ([256, 128, 64],        0.3,  0.001, 300, 25, False, "relu",  "256-128-64_d0.3_lr1e-3"),
    ([256, 128, 64],        0.3,  0.0005, 400, 30, False, "relu", "256-128-64_d0.3_lr5e-4"),
    ([256, 128, 64, 32],    0.3,  0.001, 300, 25, False, "relu",  "256-128-64-32_d0.3_lr1e-3"),
    ([256, 128, 64, 32],    0.3,  0.0005, 400, 30, False, "relu", "256-128-64-32_d0.3_lr5e-4"),
    ([128, 64, 32],         0.2,  0.001, 300, 25, True,  "relu",  "128-64-32_bn_d0.2_lr1e-3"),
    ([256, 128, 64],        0.2,  0.001, 300, 25, True,  "relu",  "256-128-64_bn_d0.2_lr1e-3"),
    ([256, 128, 64, 32],    0.3,  0.0005, 400, 30, True,  "relu", "256-128-64-32_bn_d0.3_lr5e-4"),
    ([256, 128, 64],        0.25, 0.0003, 500, 35, True,  "relu", "256-128-64_bn_d0.25_lr3e-4"),
]

best_m1 = {"acc": 0, "label": "", "model": None}
best_m2 = {"acc": 0, "label": "", "model": None}

X_m1_train, y_m1_train, X_m1_val, y_m1_val = load_data("m1")
X_m2_train, y_m2_train, X_m2_val, y_m2_val = load_data("m2")
w_m1 = compute_weights(y_m1_train)
w_m2 = compute_weights(y_m2_train)

logger.info(f"M1 train class dist: {np.bincount(y_m1_train)}, weights: {w_m1}")
logger.info(f"M2 train class dist: {np.bincount(y_m2_train)}, weights: {w_m2}")

for arch, dr, lr, epochs, patience, use_bn, activation, label in configs:
    for task, X_tr, y_tr, X_va, y_va, w, best in [
        ("M1", X_m1_train, y_m1_train, X_m1_val, y_m1_val, w_m1, best_m1),
        ("M2", X_m2_train, y_m2_train, X_m2_val, y_m2_val, w_m2, best_m2),
    ]:
        tf.random.set_seed(42)
        np.random.seed(42)
        model = build_ann(arch, dr, lr, activation, use_bn)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=epochs, batch_size=64, class_weight=w,
                      callbacks=[es], verbose=0)
        y_pred = np.argmax(model.predict(X_va, verbose=0), axis=1)
        acc = (y_pred == y_va).mean()
        loss = min(h.history["val_loss"])
        logger.info(f"  [{task}] {label}: val_acc={acc*100:.1f}%, val_loss={loss:.4f}")
        if acc > best["acc"]:
            best["acc"] = acc
            best["label"] = label

logger.info(f"\nBest M1: {best_m1['label']} @ {best_m1['acc']*100:.1f}%")
logger.info(f"Best M2: {best_m2['label']} @ {best_m2['acc']*100:.1f}%")
