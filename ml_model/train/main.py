import argparse
import logging

import pandas as pd

from ml_model.train.config import TRAIN_CSV, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", action="store_true", help="Train M1 regressor")
    parser.add_argument("--m2", action="store_true", help="Train M2 classifier")
    parser.add_argument("--data", type=str, default=TRAIN_CSV)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_rows = None
    m1_metrics = {}
    m2_metrics = {}

    df = pd.read_csv(args.data)
    data_rows = len(df)
    logger.info(f"Data: {args.data} — {data_rows} rows")

    if args.m1:
        logger.info("=== Training M1 regressor ===")
        from ml_model.train.train_m1 import train as train_m1
        from ml_model.train.evaluate import evaluate_m1
        from ml_model.train.export import save_model_m1

        model, scaler = train_m1(csv_path=args.data, split_ratio=args.split)
        m1_metrics = evaluate_m1(model, scaler, csv_path=args.data, split_ratio=args.split)
        save_model_m1(model, scaler)

    if args.m2:
        logger.info("=== Training M2 classifier ===")
        from ml_model.train.train_m2 import train as train_m2
        from ml_model.train.evaluate import evaluate_m2
        from ml_model.train.export import save_model_m2

        model, scaler, history = train_m2(csv_path=args.data, split_ratio=args.split)
        m2_metrics = evaluate_m2(model, scaler, csv_path=args.data, split_ratio=args.split)
        save_model_m2(model, scaler)

    if args.m1 or args.m2:
        from ml_model.train.export import write_model_card

        write_model_card(m1_metrics, m2_metrics, data_rows=data_rows)
        logger.info("Done. Model card written.")


if __name__ == "__main__":
    main()
