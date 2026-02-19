"""
Plot utilities.

This includes:
- plot_rmse_by_sample_features
- helper loops used in the notebook to compare configurations
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.training.modeling import train


def plot_rmse_by_sample_features(rmse_dict: Dict[str, List[Tuple[int, float]]], title: str, save_path: str | None = None, show: bool = True):
    plt.figure(figsize=(10, 6))

    for model, values in rmse_dict.items():
        if not values:
            continue
        sorted_values = sorted(values, key=lambda x: x[0])
        sample_values = [n for n, _ in sorted_values]
        rmse_values = [rmse for _, rmse in sorted_values]
        plt.plot(sample_values, rmse_values, label=model, marker="o", linestyle="-")

    if rmse_dict:
        all_samples = sorted({sample for model_values in rmse_dict.values() for sample, _ in model_values})
        plt.xticks(all_samples)

    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def obtain_comparison_RF_SVR(df_path_train: str, max_sample_cat_values: Iterable[int]):
    rmse_by_sample = {"SVR": [], "Random Forest": []}

    for max_sample in max_sample_cat_values:
        print(f"Processing max_sample_cat: {max_sample}")
        try:
            res = train(df_path_train, max_sample)
            rmse_by_sample["SVR"].append((max_sample, res.rmse_scores["Support Vector Regressor"]))
            rmse_by_sample["Random Forest"].append((max_sample, res.rmse_scores["Random Forest"]))
        except Exception as e:
            print(f"Error for max_sample_cat = {max_sample}: {e}")

    return rmse_by_sample
