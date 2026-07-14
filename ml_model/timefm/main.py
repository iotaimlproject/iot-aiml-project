import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cmd_train(args):
    from ml_model.timefm.train import train
    train(args.data)


def cmd_predict(args):
    from ml_model.timefm.predict import predictor
    predictor.load()
    import pandas as pd
    from ml_model.timefm.data import extract_batch_sequences, load_csv
    from ml_model.timefm.config import COVARIATE_CHANNELS

    df = load_csv(args.data)
    seqs = extract_batch_sequences(df)
    logger.info(f"Loaded {len(seqs)} sequences")

    results = []
    for s in seqs[:100]:
        r = predictor.predict(s["ctx_oee"].tolist(),
                              s["ctx_cov"].tolist())
        results.append(r)

    oee_preds = [r["pred_oee_10m"] for r in results]
    logger.info(f"Sample predictions (first 5): {oee_preds[:5]}")
    logger.info(f"Mean OEE: {sum(oee_preds)/len(oee_preds):.2f}")


def cmd_eval(args):
    from ml_model.timefm.predict import predictor
    predictor.load()
    import numpy as np
    from sklearn.metrics import r2_score, accuracy_score
    from ml_model.timefm.data import load_data

    (Xo_tr, Xc_tr, yo_tr, ys_tr), (Xo_va, Xc_va, yo_va, ys_va) = load_data(args.data)

    oee_preds, speed_preds = [], []
    for i in range(len(Xo_va)):
        r = predictor.predict(Xo_va[i].tolist(), Xc_va[i].tolist())
        oee_preds.append(r["pred_oee_10m"])
        speed_preds.append(r["recommended_speed"])

    from ml_model.timefm.config import SPEED_MAP
    true_speed_classes = ys_va
    pred_speed_classes = np.array([SPEED_MAP[s] for s in speed_preds])

    r2 = r2_score(yo_va[:, -1], oee_preds)
    acc = accuracy_score(true_speed_classes, pred_speed_classes)

    logger.info(f"=== Evaluation ===")
    logger.info(f"Val samples: {len(Xo_va)}")
    logger.info(f"OEE R²:      {r2:.4f}")
    logger.info(f"Speed acc:   {acc:.4f} ({acc*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeFM 2.5 OEE Forecasting")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train heads on frozen TimeFM forecasts")
    p_train.add_argument("--data", default=None)

    p_pred = sub.add_parser("predict", help="Run inference on sample data")
    p_pred.add_argument("--data", default=None)

    p_eval = sub.add_parser("eval", help="Evaluate on validation set")
    p_eval.add_argument("--data", default=None)

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
