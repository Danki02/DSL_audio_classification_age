#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.plots.rmse_plots import obtain_comparison_RF_SVR, plot_rmse_by_sample_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features_csv", required=True)
    ap.add_argument("--start", type=int, default=140)
    ap.add_argument("--stop", type=int, default=181)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    values = range(args.start, args.stop, args.step)
    rmse_dict = obtain_comparison_RF_SVR(args.train_features_csv, values)
    plot_rmse_by_sample_features(rmse_dict, "Max Samples per Category")


if __name__ == "__main__":
    main()
