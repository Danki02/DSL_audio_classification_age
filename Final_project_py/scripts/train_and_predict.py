#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from src.training.modeling import train, final_eval_process, final_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features_csv", required=True, help="CSV produced by preprocessing for training set")
    ap.add_argument("--eval_features_csv", required=True, help="CSV produced by preprocessing for evaluation set")
    ap.add_argument("--eval_ids_csv", required=False, help="Original evaluation CSV containing Id/path (to output submission in correct order)")
    ap.add_argument("--max_sample_cat", type=int, default=150)
    ap.add_argument("--out_submission", default="submission.csv")
    args = ap.parse_args()

    res = train(args.train_features_csv, args.max_sample_cat)

    eval_df = final_eval_process(args.eval_features_csv, res.x_train)

    model = final_training(res.best_model, res.train_df, res.test_df)
    preds = model.predict(eval_df)

    if args.eval_ids_csv:
        df_eval_ids = pd.read_csv(args.eval_ids_csv)
        submission = pd.DataFrame({"Id": df_eval_ids["Id"], "Predicted": preds}).sort_values(by="Id")
    else:
        submission = pd.DataFrame({"Predicted": preds})

    submission.to_csv(args.out_submission, index=False)
    print(f"Wrote: {args.out_submission}")


if __name__ == "__main__":
    main()
