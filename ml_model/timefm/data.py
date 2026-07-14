import numpy as np
import pandas as pd

from ml_model.timefm.config import (
    TRAIN_CSV,
    CONTEXT_WINDOW,
    FORECAST_HORIZON,
    OEE_CHANNEL,
    COVARIATE_CHANNELS,
    BATCH_ID,
    TIME_ORDER,
    SPEED_MAP,
    TRAIN_SPLIT,
)


def load_csv(path=None):
    path = path or TRAIN_CSV
    df = pd.read_csv(path)
    df["_speed_class"] = df["Recommended_Speed"].map(SPEED_MAP)
    return df


def extract_batch_sequences(df):
    """Group by batch, sort by position, create sliding windows."""
    sequences = []
    df = df.sort_values([BATCH_ID, TIME_ORDER]).reset_index(drop=True)

    for batch_id, group in df.groupby(BATCH_ID):
        vals = group.reset_index(drop=True)
        n = len(vals)

        if n < CONTEXT_WINDOW + 1 + FORECAST_HORIZON:
            continue

        for t in range(CONTEXT_WINDOW, n - FORECAST_HORIZON):
            ctx_oee = vals[OEE_CHANNEL].iloc[t - CONTEXT_WINDOW + 1 : t + 1].values.astype(np.float32)
            ctx_cov = vals[COVARIATE_CHANNELS].iloc[t - CONTEXT_WINDOW + 1 : t + 1].values.astype(np.float32)
            tgt_oee = vals[OEE_CHANNEL].iloc[t + 1 : t + 1 + FORECAST_HORIZON].values.astype(np.float32)
            tgt_speed = vals["_speed_class"].iloc[t]

            sequences.append({
                "batch_id": batch_id,
                "pos_ratio": vals[TIME_ORDER].iloc[t],
                "ctx_oee": ctx_oee,
                "ctx_cov": ctx_cov,
                "tgt_oee": tgt_oee,
                "tgt_speed": tgt_speed,
            })

    return sequences


def train_val_split(sequences, split_ratio=TRAIN_SPLIT):
    """Split by batch ID (alphabetical)."""
    batches = sorted(set(s["batch_id"] for s in sequences))
    split_idx = int(len(batches) * split_ratio)
    train_batches = set(batches[:split_idx])
    val_batches = set(batches[split_idx:])

    train = [s for s in sequences if s["batch_id"] in train_batches]
    val = [s for s in sequences if s["batch_id"] in val_batches]
    return train, val


def make_arrays(sequences):
    ctx_oee = np.array([s["ctx_oee"] for s in sequences], dtype=np.float32)
    ctx_cov = np.array([s["ctx_cov"] for s in sequences], dtype=np.float32)
    tgt_oee = np.array([s["tgt_oee"] for s in sequences], dtype=np.float32)
    tgt_speed = np.array([s["tgt_speed"] for s in sequences], dtype=np.int64)
    return ctx_oee, ctx_cov, tgt_oee, tgt_speed


def load_data(path=None):
    df = load_csv(path)
    seqs = extract_batch_sequences(df)
    train, val = train_val_split(seqs)
    return make_arrays(train), make_arrays(val)


if __name__ == "__main__":
    (Xo_tr, Xc_tr, yo_tr, ys_tr), (Xo_va, Xc_va, yo_va, ys_va) = load_data()
    print(f"Train: {len(Xo_tr)} windows, Val: {len(Xo_va)}")
    print(f"Context OEE shape: {Xo_tr.shape}")
    print(f"Context cov shape: {Xc_tr.shape}")
    print(f"Target OEE shape: {yo_tr.shape}")
    print(f"Target speed shape: {ys_tr.shape}")
    print(f"Speed distribution: {np.bincount(ys_tr)}")
