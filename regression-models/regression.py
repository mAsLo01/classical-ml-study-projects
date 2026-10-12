import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV


def train_regression_models(
    model_specs,
    X_train,
    X_test,
    y_train,
    y_test,
    cv,
):
    """
    Selects hyperparameters only on the training set and evaluates each
    selected model once on the held-out test set.
    """
    results = {}
    predictions = {}
    fitted_models = {}
    searches = {}

    for name, spec in model_specs.items():
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "CV RMSE": -search.best_score_,
            "Test MSE": mse,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2,
            "Best parameters": search.best_params_ or "—",
        }
        predictions[name] = y_pred
        fitted_models[name] = best_model
        searches[name] = search

    return results, predictions, fitted_models, searches


def plot_regularization(searches, output_path):
    """Plots cross-validated RMSE as a function of regularization alpha."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in ("Ridge", "Lasso"):
        search = searches[model_name]
        parameter_values = search.cv_results_["param_model__alpha"]
        alphas = np.array([float(value) for value in parameter_values])
        cv_rmse = -search.cv_results_["mean_test_score"]

        order = np.argsort(alphas)
        alphas = alphas[order]
        cv_rmse = cv_rmse[order]

        best_alpha = search.best_params_["model__alpha"]
        best_rmse = -search.best_score_

        ax.plot(alphas, cv_rmse, label=model_name)
        ax.scatter(best_alpha, best_rmse, s=70)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength (alpha)")
    ax.set_ylabel("Mean cross-validated RMSE")
    ax.set_title("Regularization parameter selection on the training set")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    tree_model,
    forest_model,
    feature_names,
    output_path,
):
    """Plots impurity-based feature importance for two tree models."""
    model_data = [
        ("Decision Tree", tree_model.feature_importances_),
        ("Random Forest", forest_model.feature_importances_),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (name, importances) in zip(axes, model_data):
        order = np.argsort(importances)
        ax.barh(
            np.asarray(feature_names)[order],
            importances[order],
        )
        ax.set_xlabel("Impurity-based importance")
        ax.set_title(name)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Feature importance (descriptive, not causal)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_diagnostics(
    y_test,
    y_pred,
    model_name,
    output_path,
):
    """Plots actual-vs-predicted values and residuals for one model."""
    residuals = y_test - y_pred
    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.75, edgecolors="black")
    axes[0].plot([lower, upper], [lower, upper], "k--")
    axes[0].set_xlabel("Actual target")
    axes[0].set_ylabel("Predicted target")
    axes[0].set_title("Actual vs predicted")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.75, edgecolors="black")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted target")
    axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residual plot")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Prediction diagnostics — {model_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_comparison(results, output_path):
    """Compares cross-validated and held-out test RMSE."""
    names = list(results)
    cv_rmse = [results[name]["CV RMSE"] for name in names]
    test_rmse = [results[name]["Test RMSE"] for name in names]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    cv_bars = ax.bar(
        x - width / 2,
        cv_rmse,
        width,
        label="Cross-validation RMSE",
    )
    test_bars = ax.bar(
        x + width / 2,
        test_rmse,
        width,
        label="Test RMSE",
    )

    ax.bar_label(cv_bars, fmt="%.1f", padding=3)
    ax.bar_label(test_bars, fmt="%.1f", padding=3)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Regression model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
