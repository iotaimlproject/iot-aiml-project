import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_LEVELS = [
    "state_NORMAL", "state_HIGH_LOAD", "state_MINOR_STOPPAGE",
    "state_MAJOR_STOPPAGE", "state_RECOVERY",
]

SPEED_SCALE_FACTOR = 10


def _classify_state(row, prev_state):
    oee_delta = row["OEE_Delta"]
    speed = row["Current_Speed"]

    if prev_state in ("MAJOR_STOPPAGE", "MINOR_STOPPAGE") and oee_delta >= 3:
        return "RECOVERY"
    if speed >= 8:
        return "HIGH_LOAD"
    if oee_delta <= -10:
        return "MAJOR_STOPPAGE"
    if oee_delta <= -4:
        return "MINOR_STOPPAGE"
    if prev_state == "RECOVERY" and abs(oee_delta) < 2:
        return "NORMAL"
    if prev_state == "HIGH_LOAD" and speed < 7:
        return "NORMAL"
    return prev_state or "NORMAL"


def _compute_target_oee_t10(df):
    """Create Predicted_OEE_t10 by shifting OEE forward 10 rows per batch."""
    df = df.copy()
    df["Predicted_OEE_t10"] = df.groupby("Batch_Part_No")["Current_OEE"].shift(-10)
    df["Predicted_OEE_t10"] = df["Predicted_OEE_t10"].fillna(df["Current_OEE"])
    return df


def enrich(df):
    df = df.copy()
    df = df.sort_values(["Batch_Part_No", "Part_SLNo"]).reset_index(drop=True)

    df["Production_End_Time"] = [
        datetime(2025, 1, 1) + timedelta(minutes=i * 5) for i in range(len(df))
    ]

    states = []
    prev = None
    for _, row in df.iterrows():
        s = _classify_state(row, prev)
        states.append(s)
        prev = s
    df["_state"] = states

    for col in STATE_LEVELS:
        df[col] = 0
    for i, state in enumerate(states):
        col = f"state_{state}"
        df.at[i, col] = 1

    df["Current_Speed"] = df["Current_Speed"] * SPEED_SCALE_FACTOR
    df["Current_Speed_pct"] = df["Current_Speed"].astype(float)
    df["Recommended_Speed"] = df["Recommended_Speed"] * SPEED_SCALE_FACTOR
    df["_pos_ratio"] = df["pos_ratio"]

    df = _compute_target_oee_t10(df)

    logger.info(
        f"Enriched: {len(df)} rows, _state distribution:\n{df['_state'].value_counts().to_string()}"
    )
    return df


def load_enriched(csv_path):
    df = pd.read_csv(csv_path)
    return enrich(df)
