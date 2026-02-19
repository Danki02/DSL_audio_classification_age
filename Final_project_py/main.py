#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.preprocessing.preprocess import preprocess_dataframe
from src.training.modeling import train, final_eval_process, final_training
from src.training.tuning import find_best_configuration
from src.plots.rmse_plots import obtain_comparison_RF_SVR, plot_rmse_by_sample_features


def run_preprocessing(csv_path: str, audio_base: str, out_csv: str, batch_size: int) -> str:
    df = pd.read_csv(csv_path)
    feats = preprocess_dataframe(df, audio_base, batch_size=batch_size)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv, index=False)
    return out_csv


def run_train_and_predict(train_features_csv: str, eval_features_csv: str, eval_ids_csv: str | None,
                          max_sample_cat: int, out_submission: str) -> str:
    res = train(train_features_csv, max_sample_cat)
    eval_df = final_eval_process(eval_features_csv, res.x_train)
    model = final_training(res.best_model, res.train_df, res.test_df)
    preds = model.predict(eval_df)

    if eval_ids_csv:
        df_eval_ids = pd.read_csv(eval_ids_csv)
        submission = pd.DataFrame({"Id": df_eval_ids["Id"], "Predicted": preds}).sort_values(by="Id")
    else:
        submission = pd.DataFrame({"Predicted": preds})

    Path(out_submission).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_submission, index=False)
    return out_submission


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end pipeline: preprocessing -> (optional) tuning -> training/prediction -> (optional) plots"
    )

    # Inputs
    ap.add_argument("--train_csv", required=True, help="Training metadata CSV")
    ap.add_argument("--eval_csv", required=True, help="Evaluation metadata CSV")
    ap.add_argument("--audio_base", required=True, help="Folder containing audio files (train/eval paths are relative to this)")
    ap.add_argument("--eval_ids_csv", default=None, help="Original eval CSV containing Id column (for ordered submission)")

    # Outputs / working dir
    ap.add_argument("--workdir", default="outputs", help="Where to write intermediate artifacts")
    ap.add_argument("--train_features_name", default="train_features.csv")
    ap.add_argument("--eval_features_name", default="eval_features.csv")
    ap.add_argument("--out_submission", default="outputs/submission.csv")

    # Preprocessing params
    ap.add_argument("--batch_size", type=int, default=100)

    # Training params
    ap.add_argument("--max_sample_cat", type=int, default=150)

    # Optional steps
    ap.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (prints best params/RMSE)")
    ap.add_argument("--tune_model", choices=["SVR", "RF"], default="RF")
    ap.add_argument("--tune_max_sample_cat", type=int, default=450)

    ap.add_argument("--make_plots", action="store_true", help="Generate RMSE comparison plot")
    ap.add_argument("--plot_start", type=int, default=140)
    ap.add_argument("--plot_stop", type=int, default=181)
    ap.add_argument("--plot_step", type=int, default=10)
    ap.add_argument("--plots_out", default=None, help="If set, saves plots to this path (e.g., outputs/rmse.png)")

    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    train_features_csv = str(workdir / args.train_features_name)
    eval_features_csv = str(workdir / args.eval_features_name)

    print("[1/4] Preprocessing train CSV -> features...")
    run_preprocessing(args.train_csv, args.audio_base, train_features_csv, args.batch_size)
    print(f"  wrote {train_features_csv}")

    print("[2/4] Preprocessing eval CSV -> features...")
    run_preprocessing(args.eval_csv, args.audio_base, eval_features_csv, args.batch_size)
    print(f"  wrote {eval_features_csv}")

    if args.tune:
        print("[optional] Hyperparameter tuning...")
        best_model, rmse, best_params = find_best_configuration(train_features_csv, args.tune_max_sample_cat, args.tune_model)
        print("  RMSE:", rmse)
        print("  Best params:", best_params)

    print("[3/4] Training + predicting...")
    out_sub = run_train_and_predict(train_features_csv, eval_features_csv, args.eval_ids_csv, args.max_sample_cat, args.out_submission)
    print(f"  wrote {out_sub}")

    if args.make_plots:
        print("[4/4] Plotting RMSE comparison...")
        values = range(args.plot_start, args.plot_stop, args.plot_step)
        rmse_dict = obtain_comparison_RF_SVR(train_features_csv, values)
        plot_rmse_by_sample_features(rmse_dict, "Max Samples per Category", save_path=args.plots_out, show=(args.plots_out is None))
        if args.plots_out:
            print(f"  wrote {args.plots_out}")


if __name__ == "__main__":
    main()
