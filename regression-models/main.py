from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from regression import (
    plot_feature_importance,
    plot_prediction_diagnostics,
    plot_regularization,
    plot_rmse_comparison,
    train_regression_models,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def build_model_specs():
    """Creates estimators and hyperparameter grids."""
    ridge_alphas = np.logspace(-4, 4, 50)
    lasso_alphas = np.logspace(-4, 2, 50)

    return {
        "Dummy Mean": {
            "estimator": DummyRegressor(strategy="mean"),
            "param_grid": {},
        },
        "Linear Regression": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            "param_grid": {},
        },
        "Ridge": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            "param_grid": {"model__alpha": ridge_alphas},
        },
        "Lasso": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=50_000)),
                ]
            ),
            "param_grid": {"model__alpha": lasso_alphas},
        },
        "Decision Tree": {
            "estimator": DecisionTreeRegressor(
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "max_depth": [2, 3, 4, 5, None],
                "min_samples_leaf": [1, 5, 10, 20],
            },
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "param_grid": {
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 3, 5],
            },
        },
    }


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    data = load_diabetes()
    X = data.data
    y = data.target

    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results, predictions, fitted_models, searches = train_regression_models(
        model_specs=build_model_specs(),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
    )

    results_table = pd.DataFrame(results).T
    results_table = results_table.sort_values("CV RMSE")

    numeric_columns = [
        "CV RMSE",
        "Test MSE",
        "Test RMSE",
        "Test MAE",
        "Test R2",
    ]
    results_table[numeric_columns] = results_table[numeric_columns].astype(float)

    print("\n" + "=" * 120)
    print("REGRESSION MODEL COMPARISON")
    print("=" * 120)
    print(results_table.round(4).to_string())

    best_model_name = results_table["CV RMSE"].idxmin()
    best_cv_rmse = results_table.loc[best_model_name, "CV RMSE"]
    best_test_rmse = results_table.loc[best_model_name, "Test RMSE"]

    print(
        "\nBest model by training-set cross-validation: "
        f"{best_model_name} "
        f"(CV RMSE = {best_cv_rmse:.3f}; "
        f"test RMSE = {best_test_rmse:.3f})"
    )

    plot_regularization(
        searches,
        FIGURES_DIR / "regularization_cv.png",
    )
    plot_feature_importance(
        fitted_models["Decision Tree"],
        fitted_models["Random Forest"],
        data.feature_names,
        FIGURES_DIR / "feature_importance.png",
    )
    plot_rmse_comparison(
        results,
        FIGURES_DIR / "rmse_comparison.png",
    )
    plot_prediction_diagnostics(
        y_test,
        predictions[best_model_name],
        best_model_name,
        FIGURES_DIR / "best_model_diagnostics.png",
    )


if __name__ == "__main__":
    main()
