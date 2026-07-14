FEATURE_CONTRACT = {
    "version": "6.0",
    "n_features": 20,
    "state_input": False,
    "speed_range": "10-100",
    "description": "20 features: 14 BASE (incl speed history) + 6 LAGS. No state.",
}

FEATURE_DEFS = [
    {"name": "Availability",           "type": "int",    "range": [0, 100],     "source": "PLC / Node-RED"},
    {"name": "Performance",            "type": "int",    "range": [0, 100],     "source": "PLC / Node-RED"},
    {"name": "Quality",                "type": "int",    "range": [0, 100],     "source": "PLC / Node-RED"},
    {"name": "Current_OEE",            "type": "int",    "range": [0, 100],     "source": "A*P*Q/10000"},
    {"name": "Current_Speed_pct",      "type": "int",    "range": [10, 100],    "source": "Speed × 10"},
    {"name": "DownTime_sec",           "type": "int",    "range": [0, None],    "source": "PLC accumulated downtime"},
    {"name": "OEE_Delta",              "type": "int",    "range": [-100, 100],  "source": "Current_OEE - Previous_OEE"},
    {"name": "Part_SLNo",              "type": "int",    "range": [1, None],    "source": "Part serial within batch"},
    {"name": "_pos_ratio",             "type": "float",  "range": [0.0, 1.0],   "source": "(Part_SLNo-1) / (total-1)"},
    {"name": "Planned_Prod_Duration",  "type": "int",    "range": [0, None],    "source": "Cumulative planned seconds"},
    {"name": "Production_Duration",    "type": "int",    "range": [0, None],    "source": "Cumulative actual seconds"},
    {"name": "Production_Delay_sec",   "type": "int",    "range": [0, None],    "source": "Actual - Planned duration"},
    {"name": "Prev_Speed_pct",         "type": "int",    "range": [10, 100],    "source": "Previous part's speed %"},
    {"name": "Speed_Delta",            "type": "int",    "range": [-90, 90],    "source": "Current - Prev speed %"},
    {"name": "OEE_lag1",               "type": "int",    "range": [0, 100],     "source": "Buffer[-2].oee"},
    {"name": "OEE_roll5_mean",         "type": "float",  "range": [0.0, 100.0], "source": "Buffer[last 5].mean"},
    {"name": "OEE_trend3",             "type": "float",  "range": [-100.0, 100.0],"source": "Buffer[-1] - Buffer[-3]"},
    {"name": "OEE_min5",               "type": "int",    "range": [0, 100],     "source": "Buffer[last 5].min"},
    {"name": "OEE_max5",               "type": "int",    "range": [0, 100],     "source": "Buffer[last 5].max"},
    {"name": "OEE_range5",             "type": "int",    "range": [0, 100],     "source": "OEE_max5 - OEE_min5"},
]

FEATURE_NAMES = [d["name"] for d in FEATURE_DEFS]


def compute_lag_features(buffer, current_oee):
    if not buffer:
        return {
            "OEE_lag1": current_oee,
            "OEE_roll5_mean": float(current_oee),
            "OEE_trend3": 0.0,
            "OEE_min5": int(current_oee),
            "OEE_max5": int(current_oee),
            "OEE_range5": 0,
        }
    vals = [r["oee"] for r in buffer]
    oee_lag1 = buffer[-1]["oee"] if len(buffer) >= 1 else current_oee
    window = vals[-5:]
    oee_roll5_mean = sum(window) / min(len(window), 5) if window else float(current_oee)
    oee_trend3 = float(window[-1] - window[0]) if len(window) >= 3 else 0.0
    oee_min5 = min(window) if window else int(current_oee)
    oee_max5 = max(window) if window else int(current_oee)
    oee_range5 = oee_max5 - oee_min5
    return {
        "OEE_lag1": oee_lag1,
        "OEE_roll5_mean": round(oee_roll5_mean, 1),
        "OEE_trend3": oee_trend3,
        "OEE_min5": oee_min5,
        "OEE_max5": oee_max5,
        "OEE_range5": oee_range5,
    }


def build_feature_vector(req_fields, lag_features):
    current_speed_pct = req_fields["current_speed"] * 10
    prev_speed_pct = req_fields.get("prev_speed_pct", current_speed_pct)
    return [
        req_fields["availability"],
        req_fields["performance"],
        req_fields["quality"],
        req_fields["current_oee"],
        current_speed_pct,
        req_fields["downtime_sec"],
        req_fields["oee_delta"],
        req_fields["part_slno"],
        req_fields["pos_ratio"],
        req_fields.get("planned_prod_duration", 0),
        req_fields.get("production_duration", 0),
        req_fields.get("production_delay_sec", 0),
        prev_speed_pct,
        current_speed_pct - prev_speed_pct,
        lag_features["OEE_lag1"],
        lag_features["OEE_roll5_mean"],
        lag_features["OEE_trend3"],
        lag_features["OEE_min5"],
        lag_features["OEE_max5"],
        lag_features["OEE_range5"],
    ]
