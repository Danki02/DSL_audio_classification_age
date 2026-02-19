"""
Training / test pipeline (as in the notebook).

Implements:
- find_best_model: compare RF vs SVR (fixed hyperparams).
- train: load feature CSV, apply per-age-bin sampling cap, train/evaluate, return artifacts.
- final_eval_process: align evaluation dataframe columns to training features.
- final_training: refit best model on train+test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class TrainResult:
    best_model: Pipeline
    x_train: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    best_rmse: float
    rmse_scores: Dict[str, float]
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def find_best_model(X_train, X_test, y_train, y_test) -> Tuple[str, float, Pipeline, Dict[str, float]]:
    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    models = {
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]),
        "Support Vector Regressor": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", SVR(kernel="rbf", C=10.769322731738809, epsilon=0.019314668917207987)),
        ]),
    }

    rmse_scores: Dict[str, float] = {}

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        rmse_scores[model_name] = rmse
        print(f"{model_name} - RMSE: {rmse}")

    best_model_name = min(rmse_scores, key=rmse_scores.get)
    best_rmse = rmse_scores[best_model_name]
    best_model = models[best_model_name]

    print(f"\nIl miglior modello è {best_model_name} con RMSE = {best_rmse}")
    return best_model_name, best_rmse, best_model, rmse_scores


def train(df_path: str, max_sample_cat: int) -> TrainResult:
    df_finale = pd.read_csv(df_path)
    train_df, test_df = train_test_split(df_finale, test_size=0.21, random_state=42)

    # Balance by age bins of width 2 years (as in notebook)
    bins = range(0, int(train_df["age"].max()) + 2, 2)
    labels = [f"{i}-{i+1}" for i in bins[:-1]]
    train_df["age_group"] = pd.cut(train_df["age"], bins=bins, labels=labels, right=False)

    def limit_samples(group):
        if len(group) > max_sample_cat:
            return group.sample(max_sample_cat, random_state=42)
        return group

    train_df = train_df.groupby("age_group", group_keys=False).apply(limit_samples)
    train_df = train_df.drop(columns=["age_group"])

    X_train = train_df.loc[:, train_df.columns.difference(["Id", "path", "age"])]
    y_train = train_df["age"]

    X_test = test_df.loc[:, test_df.columns.difference(["Id", "path", "age"])]
    y_test = test_df["age"]

    best_model_name, best_rmse, best_model, rmse_scores = find_best_model(X_train, X_test, y_train, y_test)

    return TrainResult(
        best_model=best_model,
        x_train=X_train,
        train_df=train_df,
        test_df=test_df,
        best_rmse=best_rmse,
        rmse_scores=rmse_scores,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
    )


def final_eval_process(eval_csv_path: str, x_train: pd.DataFrame) -> pd.DataFrame:
    final_eval_df = pd.read_csv(eval_csv_path)
    final_eval_df = final_eval_df.drop(columns=["Id", "path"], errors="ignore")

    train_columns = x_train.columns.tolist()
    eval_columns = final_eval_df.columns.tolist()

    missing_in_eval = set(train_columns) - set(eval_columns)
    extra_in_eval = set(eval_columns) - set(train_columns)

    print("Colonne mancanti in final_eval_df:", missing_in_eval)
    print("Colonne extra in final_eval_df:", extra_in_eval)

    # Align
    final_eval_df = final_eval_df.reindex(columns=train_columns)
    return final_eval_df


def final_training(best_model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Pipeline:
    final_training_df = pd.concat([train_df, test_df], axis=0)

    X_final = final_training_df.loc[:, final_training_df.columns.difference(["Id", "path", "age"])]
    y_final = final_training_df["age"]

    best_model.fit(X_final, y_final)
    return best_model
