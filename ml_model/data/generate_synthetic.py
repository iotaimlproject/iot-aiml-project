import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent

STATES = ["NORMAL", "HIGH_LOAD", "MINOR_STOPPAGE", "MAJOR_STOPPAGE", "RECOVERY"]

STATE_TRANSITIONS = {
    "NORMAL":           {"NORMAL": 0.90, "HIGH_LOAD": 0.04, "MINOR_STOPPAGE": 0.03, "MAJOR_STOPPAGE": 0.01, "RECOVERY": 0.02},
    "HIGH_LOAD":        {"NORMAL": 0.12, "HIGH_LOAD": 0.75, "MINOR_STOPPAGE": 0.07, "MAJOR_STOPPAGE": 0.02, "RECOVERY": 0.04},
    "MINOR_STOPPAGE":   {"NORMAL": 0.20, "HIGH_LOAD": 0.05, "MINOR_STOPPAGE": 0.58, "MAJOR_STOPPAGE": 0.07, "RECOVERY": 0.10},
    "MAJOR_STOPPAGE":   {"NORMAL": 0.08, "HIGH_LOAD": 0.02, "MINOR_STOPPAGE": 0.04, "MAJOR_STOPPAGE": 0.80, "RECOVERY": 0.06},
    "RECOVERY":         {"NORMAL": 0.55, "HIGH_LOAD": 0.10, "MINOR_STOPPAGE": 0.05, "MAJOR_STOPPAGE": 0.02, "RECOVERY": 0.28},
}

STATE_MODIFIERS = {
    "NORMAL":           {"perf": 0, "qual": 0, "avail": 0},
    "HIGH_LOAD":        {"perf": 15, "qual": -15, "avail": -8},
    "MINOR_STOPPAGE":   {"perf": -8, "qual": -5, "avail": -20},
    "MAJOR_STOPPAGE":   {"perf": -20, "qual": -12, "avail": -40},
    "RECOVERY":         {"perf": 5, "qual": 5, "avail": -3},
}

BATCH_CONFIGS = [
    {"type": "LOW",       "oee_base": 10, "degrade": 0.10, "size_range": (5, 12)},
    {"type": "MED_LOW",   "oee_base": 25, "degrade": 0.07, "size_range": (5, 12)},
    {"type": "MED",       "oee_base": 45, "degrade": 0.06, "size_range": (5, 12)},
    {"type": "MED_HIGH",  "oee_base": 65, "degrade": 0.05, "size_range": (5, 12)},
    {"type": "HIGH",      "oee_base": 80, "degrade": 0.03, "size_range": (5, 12)},
    {"type": "TOP",       "oee_base": 92, "degrade": 0.02, "size_range": (5, 12)},
]

SPEED_LEVELS = list(range(1, 11))
SPEED_PCT = {s: s * 10 for s in SPEED_LEVELS}

SPEED_PERF_BOOST      = {1: 0, 2: 4, 3: 9, 4: 15, 5: 22, 6: 30, 7: 39, 8: 49, 9: 60, 10: 72}
SPEED_PENALTY_QUALITY = {1: 0, 2: 0, 3: 1, 4: 4, 5: 8, 6: 14, 7: 21, 8: 30, 9: 40, 10: 52}
SPEED_PENALTY_AVAIL   = {1: 0, 2: 0, 3: 0, 4: 1, 5: 3, 6: 6, 7: 10, 8: 15, 9: 22, 10: 30}

PERF_BASE = 42
QUAL_BASE = 92
AVAIL_BASE = 82

RECOMMENDED_SPEED_MAP = {1: 20, 2: 20, 3: 40, 4: 40, 5: 60, 6: 60, 7: 80, 8: 80, 9: 100, 10: 100}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pick_next_state(state):
    probs = STATE_TRANSITIONS[state]
    return random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]


def compute_oee_components(part_slno, total_size, speed, state, oee_base, degrade, noise_scale=0.3):
    """Compute steady-state A/P/Q/OEE for a given speed. No lag effect."""
    pos_ratio = (part_slno - 1) / max(total_size - 1, 1)

    mod = STATE_MODIFIERS[state]
    scale = 0.4 + (oee_base / 50.0) * 0.6

    perf_raw = PERF_BASE * scale + SPEED_PERF_BOOST[speed] + mod["perf"] - pos_ratio * 8
    qual_raw = QUAL_BASE * scale - SPEED_PENALTY_QUALITY[speed] + mod["qual"] - pos_ratio * 3
    avail_raw = AVAIL_BASE * scale - SPEED_PENALTY_AVAIL[speed] + mod["avail"] - pos_ratio * 3

    noise = np.random.normal(0, noise_scale, 3)
    perf = clamp(perf_raw + noise[0], 0, 100)
    qual = clamp(qual_raw + noise[1], 0, 100)
    avail = clamp(avail_raw + noise[2], 0, 100)

    oee = int(round((avail * perf * qual) / 10000))
    oee = clamp(oee, 0, 100)
    return int(round(avail)), int(round(perf)), int(round(qual)), oee


def find_optimal_speed(part_slno, total_size, state, oee_base, degrade):
    candidates = []
    for s in SPEED_LEVELS:
        _, _, _, oee = compute_oee_components(part_slno, total_size, s, state, oee_base, degrade, noise_scale=0)
        candidates.append((s, oee))
    candidates.sort(key=lambda x: -x[1])
    best_oee = candidates[0][1]
    top_tier = [s for s, o in candidates if o >= best_oee - 1.5]
    return random.choice(top_tier)


def inject_crash_events(rows, crash_prob=0.035):
    rows_by_batch = {}
    for i, r in enumerate(rows):
        rows_by_batch.setdefault(r["Batch_PartNo"], []).append(i)

    modified = set()
    crashes = 0
    for batch_id, indices in rows_by_batch.items():
        if len(indices) < 8:
            continue

        if random.random() > crash_prob * len(indices):
            continue

        crash_start = indices[random.randint(0, max(0, len(indices) - 6))]
        crash_len = random.randint(2, 4)
        recover_len = random.randint(2, 5)

        for j in range(crash_len):
            idx = crash_start + j
            if idx >= len(rows):
                break
            modified.add(idx)
            r = rows[idx]
            depth = max(0.15, 1.0 - (j / max(crash_len, 1)) * 0.8)
            r["Availability"] = max(0, int(r["Availability"] * depth * random.uniform(0.3, 0.6)))
            r["Performance"] = max(0, int(r["Performance"] * depth * random.uniform(0.3, 0.6)))
            r["Quality"] = max(0, int(r["Quality"] * depth * random.uniform(0.4, 0.7)))
            r["Current_OEE"] = max(0, int((r["Availability"] * r["Performance"] * r["Quality"]) / 10000))
            r["DownTime_sec"] = random.randint(60, 300)

        for j in range(recover_len):
            idx = crash_start + crash_len + j
            if idx >= len(rows):
                break
            modified.add(idx)
            r = rows[idx]
            t = min(1.0, (j + 1) / recover_len)
            r["Availability"] = min(100, max(0, int(r["Availability"] + (82 - r["Availability"]) * t * 0.5)))
            r["Performance"] = min(100, max(0, int(r["Performance"] + (65 - r["Performance"]) * t * 0.5)))
            r["Quality"] = min(100, max(0, int(r["Quality"] + (85 - r["Quality"]) * t * 0.5)))
            r["Current_OEE"] = min(100, max(0, int((r["Availability"] * r["Performance"] * r["Quality"]) / 10000)))
            r["DownTime_sec"] = max(0, int(r["DownTime_sec"] * 0.5))

        crashes += 1

    print(f"  Injected {crashes} crash events (modified {len(modified)} rows)")

    for i in modified:
        oee = rows[i]["Current_OEE"]
        rows[i]["Recommended_Speed"] = 20 if oee < 8 else (40 if oee < 20 else (60 if oee < 40 else 80))
        rows[i]["OEE_Delta"] = rows[i]["Current_OEE"] - (rows[i-1]["Current_OEE"] if i > 0 else 0)
        if i + 1 < len(rows) and i + 1 not in modified:
            rows[i + 1]["OEE_Delta"] = rows[i + 1]["Current_OEE"] - rows[i]["Current_OEE"]

    for batch_id in rows_by_batch:
        indices = rows_by_batch[batch_id]
        cum_actual = 0
        cum_planned = 0
        planned_per_part = 60
        for idx in indices:
            r = rows[idx]
            cum_planned += planned_per_part
            base_dur = 75
            speed_pct = r["Current_Speed_pct"]
            speed_factor = max(speed_pct, 10) / 80.0
            cum_actual += int(base_dur * speed_factor) + r.get("DownTime_sec", 0)
            r["Planned_Prod_Duration"] = cum_planned
            r["Production_Duration"] = cum_actual
            r["Production_Delay_sec"] = max(0, cum_actual - cum_planned)

    return rows


def generate_dataset(n_batches=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    batch_counter = 0
    base_time = datetime(2026, 1, 4, 19, 0, 0)
    time_cursor = base_time

    while batch_counter < n_batches and len(rows) < 80000:
        cfg = random.choice(BATCH_CONFIGS)
        batch_type = cfg["type"]
        oee_base = cfg["oee_base"]
        degrade = cfg["degrade"]
        total_size = random.randint(*cfg["size_range"])
        batch_counter += 1
        batch_id = f"{batch_type}_b{batch_counter}"

        state = "NORMAL"

        current_speed = random.choice(SPEED_LEVELS)
        effective_speed = current_speed
        prev_speed_at_change = current_speed
        lag_progress = 999

        prev_oee = None
        cum_planned = 0
        cum_actual = 0
        planned_per_part = 60
        speed_change_counter = 0
        speed_change_interval = random.randint(2, 4)

        prev_speed_pct = SPEED_PCT[current_speed]

        for part_slno in range(1, total_size + 1):
            state = pick_next_state(state)

            speed_change_counter += 1
            if speed_change_counter >= speed_change_interval and part_slno < total_size - 2:
                prev_speed_at_change = current_speed
                current_speed = random.choice(SPEED_LEVELS)
                speed_change_counter = 0
                speed_change_interval = random.randint(2, 4)
                lag_progress = 0
            else:
                if lag_progress < 999:
                    lag_progress += 1

            if lag_progress >= 4:
                effective_speed = current_speed
            elif lag_progress == 0:
                effective_speed = prev_speed_at_change
            elif lag_progress == 1:
                effective_speed = int(prev_speed_at_change * 0.75 + current_speed * 0.25)
            elif lag_progress == 2:
                effective_speed = int(prev_speed_at_change * 0.50 + current_speed * 0.50)
            elif lag_progress == 3:
                effective_speed = int(prev_speed_at_change * 0.25 + current_speed * 0.75)

            avail, perf, qual, oee = compute_oee_components(
                part_slno, total_size, effective_speed, state, oee_base, degrade,
            )

            oee_delta = (oee - prev_oee) if prev_oee is not None else 0
            prev_oee = oee

            if state == "MAJOR_STOPPAGE":
                downtime_sec = random.randint(120, 600)
            elif state == "MINOR_STOPPAGE":
                downtime_sec = random.randint(30, 180)
            else:
                downtime_sec = 0

            optimal_speed = find_optimal_speed(part_slno, total_size, state, oee_base, degrade)
            optimal_speed_pct = RECOMMENDED_SPEED_MAP[optimal_speed]

            pos_ratio = round((part_slno - 1) / max(total_size - 1, 1), 4)

            base_duration = random.randint(60, 90)
            time_cursor += timedelta(seconds=base_duration)

            cum_planned += planned_per_part
            speed_factor = max(SPEED_PCT[current_speed], 10) / 80.0
            cum_actual += int(base_duration * speed_factor) + downtime_sec
            prod_delay = max(0, cum_actual - cum_planned)

            current_speed_pct = SPEED_PCT[current_speed]
            speed_delta = current_speed_pct - prev_speed_pct

            part_no = f"{batch_id}_{part_slno}"

            rows.append({
                "Production_End_Time": time_cursor.strftime("%Y-%m-%d %H:%M:%S"),
                "Batch_PartNo": batch_id,
                "Part_No": part_no,
                "Part_SLNo": part_slno,
                "Availability": avail,
                "Performance": perf,
                "Quality": qual,
                "Current_OEE": oee,
                "Current_Speed_pct": current_speed_pct,
                "Prev_Speed_pct": prev_speed_pct,
                "Speed_Delta": speed_delta,
                "DownTime_sec": downtime_sec,
                "OEE_Delta": int(oee_delta),
                "_pos_ratio": pos_ratio,
                "Planned_Prod_Duration": cum_planned,
                "Production_Duration": cum_actual,
                "Production_Delay_sec": prod_delay,
                "Recommended_Speed": optimal_speed_pct,
            })

            prev_speed_pct = current_speed_pct

            if len(rows) >= 80000:
                break

    return rows


def compute_lag_and_target(rows):
    result = []
    current_batch = None
    batch_oee_hist = []
    all_batch_oee_by_batch = {}

    for r in rows:
        if r["Batch_PartNo"] not in all_batch_oee_by_batch:
            all_batch_oee_by_batch[r["Batch_PartNo"]] = [
                x["Current_OEE"] for x in rows if x["Batch_PartNo"] == r["Batch_PartNo"]
            ]

    for r in rows:
        batch_id = r["Batch_PartNo"]
        if batch_id != current_batch:
            current_batch = batch_id
            batch_oee_hist = []

        batch_oee_hist.append(r["Current_OEE"])

        oee_lag1 = batch_oee_hist[-2] if len(batch_oee_hist) >= 2 else r["Current_OEE"]
        window = batch_oee_hist[-5:]
        oee_roll5_mean = round(sum(window) / len(window), 1)
        oee_trend3 = float(window[-1] - window[0]) if len(window) >= 3 else 0.0
        oee_min5 = min(window)
        oee_max5 = max(window)
        oee_range5 = oee_max5 - oee_min5

        all_oee = all_batch_oee_by_batch[batch_id]
        idx_in_batch = len(batch_oee_hist) - 1
        target_idx = min(idx_in_batch + 1, len(all_oee) - 1)
        pred_oee_t1 = all_oee[target_idx]

        rr = dict(r)
        rr["OEE_lag1"] = oee_lag1
        rr["OEE_roll5_mean"] = oee_roll5_mean
        rr["OEE_trend3"] = oee_trend3
        rr["OEE_min5"] = oee_min5
        rr["OEE_max5"] = oee_max5
        rr["OEE_range5"] = oee_range5
        rr["Predicted_OEE_t1"] = pred_oee_t1
        result.append(rr)

    return result


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = [
        "Production_End_Time", "Batch_PartNo", "Part_No", "Part_SLNo",
        "Availability", "Performance", "Quality", "Current_OEE",
        "Current_Speed_pct", "Prev_Speed_pct", "Speed_Delta",
        "DownTime_sec", "OEE_Delta",
        "_pos_ratio",
        "Planned_Prod_Duration", "Production_Duration", "Production_Delay_sec",
        "Predicted_OEE_t1", "Recommended_Speed",
        "OEE_lag1", "OEE_roll5_mean", "OEE_trend3",
        "OEE_min5", "OEE_max5", "OEE_range5",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "syn_oee_10k.csv"))
    args = parser.parse_args()

    print(f"Generating up to {args.n_batches} batches...")
    raw = generate_dataset(n_batches=args.n_batches, seed=args.seed)
    print(f"Raw rows: {len(raw)}")

    print("Injecting crash events...")
    with_crashes = inject_crash_events(raw)

    with_targets = compute_lag_and_target(with_crashes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(with_targets, str(out_path))

    oee_vals = [r["Current_OEE"] for r in with_targets]
    print(f"  OEE range: [{min(oee_vals)}, {max(oee_vals)}], mean={sum(oee_vals)/len(oee_vals):.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
