#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.training.tuning import find_best_configuration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features_csv", required=True)
    ap.add_argument("--max_sample_cat", type=int, default=450)
    ap.add_argument("--model", choices=["SVR", "RF"], default="RF")
    args = ap.parse_args()

    best_model, rmse, best_params = find_best_configuration(args.train_features_csv, args.max_sample_cat, args.model)
    print("RMSE:", rmse)
    print("Best params:", best_params)


if __name__ == "__main__":
    main()
