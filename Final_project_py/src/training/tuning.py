"""
Hyperparameter tuning utilities (Optuna for SVR, GridSearch for RF),
ported from the notebook.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def optuna_best_model(X_train, X_test, y_train, y_test) -> Tuple[Pipeline, float, Dict[str, Any]]:
    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf"])
        C = trial.suggest_float("C", 10, 11, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.02, log=True)

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", SVR(kernel=kernel, C=C, epsilon=epsilon)),
        ])

        score = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print(f"Migliori iperparametri trovati: {best_params}")

    best_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", SVR(kernel=best_params["kernel"], C=best_params["C"], epsilon=best_params["epsilon"])),
    ])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"Miglior modello RMSE sul test set: {rmse}")
    return best_model, rmse, best_params


def grid_search_random_forest(X_train, X_test, y_train, y_test) -> Tuple[Pipeline, float, Dict[str, Any]]:
    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ])

    param_grid = {
        "regressor__n_estimators": [100, 300],
        "regressor__max_depth": [5, 10, None],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Migliori parametri trovati: {best_params}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"Miglior modello RMSE sul test set: {rmse}")
    return best_model, rmse, best_params


def find_best_configuration(df_path: str, max_sample_cat: int, model_name: str):
    df_finale = pd.read_csv(df_path)
    train_df, test_df = train_test_split(df_finale, test_size=0.21, random_state=42)

    bins = range(0, int(train_df["age"].max()) + 2, 2)
    labels = [f"{i}-{i+1}" for i in bins[:-1]]
    train_df["age_group"] = pd.cut(train_df["age"], bins=bins, labels=labels, right=False)

    def limit_samples(group):
        if len(group) > max_sample_cat:
            return group.sample(max_sample_cat, random_state=42)
        return group

    train_df = train_df.groupby("age_group", group_keys=False).apply(limit_samples)
    train_df = train_df.drop(columns=["age_group"])

    x_train = train_df.loc[:, train_df.columns.difference(["Id", "path", "age"])]
    y_train = train_df["age"]
    x_test = test_df.loc[:, test_df.columns.difference(["Id", "path", "age"])]
    y_test = test_df["age"]

    model_name = model_name.upper()
    if model_name == "SVR":
        return optuna_best_model(x_train, x_test, y_train, y_test)
    if model_name == "RF":
        return grid_search_random_forest(x_train, x_test, y_train, y_test)
    raise ValueError("model_name must be 'SVR' or 'RF'")
