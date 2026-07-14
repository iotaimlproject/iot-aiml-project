import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from ml_model.train.config import FEATURE_NAMES, TRAIN_CSV, DIRECTION_MAP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N = len(FEATURE_NAMES)

def load_m2(split_ratio=0.8):
    df = pd.read_csv(TRAIN_CSV).sort_values(["Batch_Part_No", "Part_SLNo"]).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    y_train_3 = np.array([DIRECTION_MAP[v] for v in train_df["Speed_Direction"].values])
    y_val_3 = np.array([DIRECTION_MAP[v] for v in val_df["Speed_Direction"].values])
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    return X_tr_s, y_train_3, X_va_s, y_val_3

def try_two_stage():
    logger.info("\n=== M2 TWO-STAGE ANN ===")
    X_train, y_train_3, X_val, y_val_3 = load_m2()
    y_tr_s1 = (y_train_3 != 1).astype(int)
    y_va_s1 = (y_val_3 != 1).astype(int)

    # Stage 1: change (0/1/2) vs hold (1)
    tf.random.set_seed(42)
    np.random.seed(42)
    s1 = Sequential([InputLayer(shape=(N,)),
        Dense(64, activation="relu"), Dropout(0.2),
        Dense(32, activation="relu"), Dropout(0.1),
        Dense(1, activation="sigmoid")])
    s1.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    w_s1 = {0: 1.0, 1: 5.0}
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    s1.fit(X_train, y_tr_s1, validation_data=(X_val, y_va_s1), epochs=200, batch_size=64,
           class_weight=w_s1, callbacks=[es], verbose=0)
    s1_acc = (s1.predict(X_val, verbose=0).ravel() > 0.5).astype(int)
    s1_acc = (s1_acc == y_va_s1).mean()
    logger.info(f"  Stage 1 (change vs hold): {s1_acc*100:.1f}%")

    # Stage 2: up (2) vs down (0)
    cm_tr = y_train_3 != 1
    cm_va = y_val_3 != 1
    y_tr_s2 = (y_train_3[cm_tr] == 2).astype(int)
    y_va_s2 = (y_val_3[cm_va] == 2).astype(int)
    tf.random.set_seed(42)
    np.random.seed(42)
    s2 = Sequential([InputLayer(shape=(N,)),
        Dense(64, activation="relu"), Dropout(0.2),
        Dense(32, activation="relu"), Dropout(0.1),
        Dense(1, activation="sigmoid")])
    s2.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    s2.fit(X_train[cm_tr], y_tr_s2, validation_data=(X_val[cm_va], y_va_s2),
           epochs=200, batch_size=64, class_weight="balanced",
           callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)], verbose=0)
    s2_acc = (s2.predict(X_val[cm_va], verbose=0).ravel() > 0.5).astype(int)
    s2_acc = (s2_acc == y_va_s2).mean()
    logger.info(f"  Stage 2 (up vs down): {s2_acc*100:.1f}%")

    # Combined
    s1_p = (s1.predict(X_val, verbose=0).ravel() > 0.5).astype(int)
    combined = np.full(len(y_val_3), 1, dtype=int)
    change = s1_p == 1
    if change.any():
        s2_p = (s2.predict(X_val[change], verbose=0).ravel() > 0.5).astype(int)
        combined[change] = np.where(s2_p == 0, 0, 2)
    acc = (combined == y_val_3).mean()
    logger.info(f"  Combined: {acc*100:.1f}%")
    return {"s1": s1, "s2": s2}, combined, y_val_3

def try_deep_3class():
    logger.info("\n=== M2 DEEPER 3-CLASS ===")
    X_train, y_train_3, X_val, y_val_3 = load_m2()
    w = {0: 0.78, 1: 3.45, 2: 0.70}

    tf.random.set_seed(42)
    np.random.seed(42)
    model = Sequential([InputLayer(shape=(N,)),
        Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),
        Dense(256), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),
        Dense(128), BatchNormalization(), LeakyReLU(0.1), Dropout(0.2),
        Dense(64), LeakyReLU(0.1), Dropout(0.1),
        Dense(3, activation="softmax")])
    model.compile(optimizer=Adam(0.0003), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    model.fit(X_train, y_train_3, validation_data=(X_val, y_val_3),
              epochs=500, batch_size=128, class_weight=w, callbacks=[es], verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = (y_pred == y_val_3).mean()
    logger.info(f"  Accuracy: {acc*100:.1f}%")

    # Per-class
    for cls, name in [(0,"down"),(1,"hold"),(2,"up")]:
        mask = y_val_3 == cls
        if mask.any():
            ca = (y_pred[mask] == cls).mean()
            logger.info(f"    {name}: {ca*100:.1f}%")
    return model

def try_sgd():
    logger.info("\n=== M2 SGD WITH MOMENTUM ===")
    X_train, y_train_3, X_val, y_val_3 = load_m2()
    w = {0: 0.78, 1: 3.45, 2: 0.70}

    tf.random.set_seed(42)
    np.random.seed(42)
    model = Sequential([InputLayer(shape=(N,)),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(64, activation="relu"), Dropout(0.1),
        Dense(3, activation="softmax")])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    model.fit(X_train, y_train_3, validation_data=(X_val, y_val_3),
              epochs=500, batch_size=128, class_weight=w, callbacks=[es], verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = (y_pred == y_val_3).mean()
    logger.info(f"  Accuracy: {acc*100:.1f}%")
    for cls, name in [(0,"down"),(1,"hold"),(2,"up")]:
        mask = y_val_3 == cls
        if mask.any():
            ca = (y_pred[mask] == cls).mean()
            logger.info(f"    {name}: {ca*100:.1f}%")
    return model

two_stage_result, combined_pred, y_true = try_two_stage()
logger.info("")
try_deep_3class()
logger.info("")
try_sgd()
