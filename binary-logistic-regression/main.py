from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mylogclass import mylearn, mypredict, mypredict_proba


RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_INDICES = [2, 3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def print_metrics(model_name, y_true, y_pred):
    """Prints classification metrics for one model."""
    print(f"\n{model_name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def make_grid(X, margin=0.5, points_per_axis=200):
    """Creates a two-dimensional grid in the original feature scale."""
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, points_per_axis),
        np.linspace(y_min, y_max, points_per_axis),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    return xx, yy, grid


def plot_probability_surface(
    xx,
    yy,
    probability_surface,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    title,
    filename,
):
    """Plots a probability surface and the train/test observations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(
        xx,
        yy,
        probability_surface,
        levels=np.linspace(0.0, 1.0, 11),
        alpha=0.35,
        cmap="RdYlBu",
    )
    ax.contour(
        xx,
        yy,
        probability_surface,
        levels=[0.5],
        colors="black",
        linewidths=2,
    )

    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="RdYlBu",
        marker="o",
        edgecolors="black",
        alpha=0.75,
        label="Train",
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="RdYlBu",
        marker="X",
        edgecolors="black",
        s=90,
        label="Test",
    )

    fig.colorbar(contour, ax=ax, label="P(virginica)")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(losses2, losses4):
    """Plots binary cross-entropy during gradient descent."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses2, label="2 features")
    ax.plot(losses4, label="4 features")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Training loss of custom logistic regression")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "training_loss.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    data = load_iris()
    X = data.data
    y = (data.target == 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_names = [data.feature_names[index] for index in FEATURE_INDICES]

    # ------------------------------------------------------------------
    # Experiment 1: petal length and petal width
    # ------------------------------------------------------------------
    X2_train_raw = X_train[:, FEATURE_INDICES]
    X2_test_raw = X_test[:, FEATURE_INDICES]
    X2_raw = X[:, FEATURE_INDICES]

    scaler2 = StandardScaler()
    X2_train = scaler2.fit_transform(X2_train_raw)
    X2_test = scaler2.transform(X2_test_raw)

    theta2, losses2 = mylearn(
        X2_train,
        y_train,
        learning_rate=0.1,
        num_iterations=5000,
    )
    y_pred_custom2 = mypredict(theta2, X2_test)

    sklearn2 = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000,
    )
    sklearn2.fit(X2_train, y_train)
    y_pred_sklearn2 = sklearn2.predict(X2_test)

    print_metrics(
        "Custom logistic regression — 2 features",
        y_test,
        y_pred_custom2,
    )
    print_metrics(
        "Scikit-learn logistic regression — 2 features",
        y_test,
        y_pred_sklearn2,
    )

    xx, yy, grid2_raw = make_grid(X2_raw)
    grid2 = scaler2.transform(grid2_raw)

    custom_surface2 = mypredict_proba(theta2, grid2).reshape(xx.shape)
    sklearn_surface2 = sklearn2.predict_proba(grid2)[:, 1].reshape(xx.shape)

    plot_probability_surface(
        xx,
        yy,
        custom_surface2,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names,
        "Custom logistic regression — 2 features",
        "custom_logistic_regression_2_features.png",
    )
    plot_probability_surface(
        xx,
        yy,
        sklearn_surface2,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names,
        "Scikit-learn logistic regression — 2 features",
        "sklearn_logistic_regression_2_features.png",
    )

    # ------------------------------------------------------------------
    # Experiment 2: all four Iris features
    # ------------------------------------------------------------------
    scaler4 = StandardScaler()
    X4_train = scaler4.fit_transform(X_train)
    X4_test = scaler4.transform(X_test)

    theta4, losses4 = mylearn(
        X4_train,
        y_train,
        learning_rate=0.1,
        num_iterations=5000,
    )
    y_pred_custom4 = mypredict(theta4, X4_test)

    sklearn4 = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000,
    )
    sklearn4.fit(X4_train, y_train)
    y_pred_sklearn4 = sklearn4.predict(X4_test)

    print_metrics(
        "Custom logistic regression — 4 features",
        y_test,
        y_pred_custom4,
    )
    print_metrics(
        "Scikit-learn logistic regression — 4 features",
        y_test,
        y_pred_sklearn4,
    )

    # The two unused features are fixed at their training-set means.
    # Therefore, this is a 2D slice of the 4D decision surface, not a
    # mathematical projection of the full feature space.
    train_means = X_train.mean(axis=0)
    grid4_raw = np.tile(train_means, (grid2_raw.shape[0], 1))
    grid4_raw[:, FEATURE_INDICES] = grid2_raw
    grid4 = scaler4.transform(grid4_raw)

    custom_surface4 = mypredict_proba(theta4, grid4).reshape(xx.shape)
    sklearn_surface4 = sklearn4.predict_proba(grid4)[:, 1].reshape(xx.shape)

    plot_probability_surface(
        xx,
        yy,
        custom_surface4,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names,
        "Custom logistic regression — 4 features\n"
        "2D slice at mean sepal values",
        "custom_logistic_regression_4_features_slice.png",
    )
    plot_probability_surface(
        xx,
        yy,
        sklearn_surface4,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names,
        "Scikit-learn logistic regression — 4 features\n"
        "2D slice at mean sepal values",
        "sklearn_logistic_regression_4_features_slice.png",
    )

    plot_training_loss(losses2, losses4)


if __name__ == "__main__":
    main()
