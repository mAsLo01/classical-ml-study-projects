from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mymulticlass import mylearn, mypredict


RANDOM_STATE = 42
TEST_SIZE = 0.2
IRIS_FEATURE_INDICES = [2, 3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

CLASS_COLORS = ["#d73027", "#fee090", "#4575b4"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)


def print_metrics(model_name, y_true, y_pred, target_names):
    """Prints test metrics for a multiclass classifier."""
    print(f"\n{model_name}")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.3f}")
    print(
        "Macro precision: "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.3f}"
    )
    print(
        "Macro recall:    "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.3f}"
    )
    print(
        "Macro F1-score:  "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=3,
            zero_division=0,
        )
    )


def make_grid(X, margin=0.5, points_per_axis=250):
    """Creates a two-dimensional grid around plotted observations."""
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, points_per_axis),
        np.linspace(y_min, y_max, points_per_axis),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, grid


def plot_decision_regions(
    xx,
    yy,
    predicted_classes,
    X_train_plot,
    y_train,
    X_test_plot,
    y_test,
    xlabel,
    ylabel,
    title,
    filename,
    n_classes=3,
):
    """Plots decision regions with separate train and test markers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.arange(n_classes + 1) - 0.5

    ax.contourf(
        xx,
        yy,
        predicted_classes,
        levels=levels,
        alpha=0.3,
        cmap=CLASS_CMAP,
    )
    ax.contour(
        xx,
        yy,
        predicted_classes,
        levels=np.arange(n_classes - 1) + 0.5,
        colors="black",
        linewidths=1,
    )

    ax.scatter(
        X_train_plot[:, 0],
        X_train_plot[:, 1],
        c=y_train,
        cmap=CLASS_CMAP,
        marker="o",
        edgecolors="black",
        alpha=0.75,
        label="Train",
    )
    ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test,
        cmap=CLASS_CMAP,
        marker="X",
        edgecolors="black",
        s=90,
        label="Test",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_losses(losses):
    """Plots cross-entropy histories of the custom classifiers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in losses.items():
        ax.plot(values, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy")
    ax.set_title("Training loss of custom softmax classifiers")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "training_losses.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_iris_experiments():
    """Runs two- and four-feature experiments on Iris."""
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Iris: two petal features
    # ------------------------------------------------------------------
    X2_train_raw = X_train[:, IRIS_FEATURE_INDICES]
    X2_test_raw = X_test[:, IRIS_FEATURE_INDICES]
    X2_raw = X[:, IRIS_FEATURE_INDICES]

    scaler2 = StandardScaler()
    X2_train = scaler2.fit_transform(X2_train_raw)
    X2_test = scaler2.transform(X2_test_raw)

    weights2, losses2 = mylearn(
        X2_train,
        y_train,
        learning_rate=0.1,
        num_iterations=5000,
    )
    custom_pred2 = mypredict(weights2, X2_test)

    sklearn2 = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000,
    )
    sklearn2.fit(X2_train, y_train)
    sklearn_pred2 = sklearn2.predict(X2_test)

    print_metrics(
        "Custom softmax — Iris, 2 features",
        y_test,
        custom_pred2,
        data.target_names,
    )
    print_metrics(
        "Scikit-learn — Iris, 2 features",
        y_test,
        sklearn_pred2,
        data.target_names,
    )

    xx, yy, grid2_raw = make_grid(X2_raw)
    grid2 = scaler2.transform(grid2_raw)

    custom_regions2 = mypredict(weights2, grid2).reshape(xx.shape)
    sklearn_regions2 = sklearn2.predict(grid2).reshape(xx.shape)

    feature_names = [
        data.feature_names[index] for index in IRIS_FEATURE_INDICES
    ]

    plot_decision_regions(
        xx,
        yy,
        custom_regions2,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names[0],
        feature_names[1],
        "Custom softmax — Iris, 2 features",
        "iris_custom_2_features.png",
    )
    plot_decision_regions(
        xx,
        yy,
        sklearn_regions2,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names[0],
        feature_names[1],
        "Scikit-learn logistic regression — Iris, 2 features",
        "iris_sklearn_2_features.png",
    )

    # ------------------------------------------------------------------
    # Iris: all four features
    # ------------------------------------------------------------------
    scaler4 = StandardScaler()
    X4_train = scaler4.fit_transform(X_train)
    X4_test = scaler4.transform(X_test)

    weights4, losses4 = mylearn(
        X4_train,
        y_train,
        learning_rate=0.1,
        num_iterations=5000,
    )
    custom_pred4 = mypredict(weights4, X4_test)

    sklearn4 = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000,
    )
    sklearn4.fit(X4_train, y_train)
    sklearn_pred4 = sklearn4.predict(X4_test)

    print_metrics(
        "Custom softmax — Iris, 4 features",
        y_test,
        custom_pred4,
        data.target_names,
    )
    print_metrics(
        "Scikit-learn — Iris, 4 features",
        y_test,
        sklearn_pred4,
        data.target_names,
    )

    # The unused sepal features are fixed at their training-set means.
    # This is a 2D slice of the 4D decision surface, not a projection.
    train_means = X_train.mean(axis=0)
    grid4_raw = np.tile(train_means, (grid2_raw.shape[0], 1))
    grid4_raw[:, IRIS_FEATURE_INDICES] = grid2_raw
    grid4 = scaler4.transform(grid4_raw)

    custom_regions4 = mypredict(weights4, grid4).reshape(xx.shape)
    sklearn_regions4 = sklearn4.predict(grid4).reshape(xx.shape)

    plot_decision_regions(
        xx,
        yy,
        custom_regions4,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names[0],
        feature_names[1],
        "Custom softmax — Iris, 4 features\n"
        "2D slice at mean sepal values",
        "iris_custom_4_features_slice.png",
    )
    plot_decision_regions(
        xx,
        yy,
        sklearn_regions4,
        X2_train_raw,
        y_train,
        X2_test_raw,
        y_test,
        feature_names[0],
        feature_names[1],
        "Scikit-learn logistic regression — Iris, 4 features\n"
        "2D slice at mean sepal values",
        "iris_sklearn_4_features_slice.png",
    )

    return {
        "Iris — 2 features": losses2,
        "Iris — 4 features": losses4,
    }


def run_wine_experiment():
    """Runs full-feature classification with PCA used for visualization."""
    data = load_wine()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weights, losses = mylearn(
        X_train_scaled,
        y_train,
        learning_rate=0.1,
        num_iterations=5000,
    )
    custom_pred = mypredict(weights, X_test_scaled)

    sklearn_model = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000,
    )
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_model.predict(X_test_scaled)

    print_metrics(
        "Custom softmax — Wine, 13 features",
        y_test,
        custom_pred,
        data.target_names,
    )
    print_metrics(
        "Scikit-learn — Wine, 13 features",
        y_test,
        sklearn_pred,
        data.target_names,
    )

    # PCA is fitted only on training data and is used only to visualize
    # the full-feature classifiers on the first two principal components.
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_pca_for_grid = np.vstack((X_train_pca, X_test_pca))
    xx, yy, grid_pca = make_grid(X_pca_for_grid, margin=1.0)

    # inverse_transform reconstructs points in the standardized
    # 13-dimensional feature space with the omitted components set to zero.
    grid_scaled = pca.inverse_transform(grid_pca)

    custom_regions = mypredict(weights, grid_scaled).reshape(xx.shape)
    sklearn_regions = sklearn_model.predict(grid_scaled).reshape(xx.shape)

    explained = pca.explained_variance_ratio_ * 100
    pc1_label = f"PC1 ({explained[0]:.1f}% variance)"
    pc2_label = f"PC2 ({explained[1]:.1f}% variance)"

    plot_decision_regions(
        xx,
        yy,
        custom_regions,
        X_train_pca,
        y_train,
        X_test_pca,
        y_test,
        pc1_label,
        pc2_label,
        "Custom softmax — Wine, 13 features\n"
        "Decision regions on the first two principal components",
        "wine_custom_pca_slice.png",
    )
    plot_decision_regions(
        xx,
        yy,
        sklearn_regions,
        X_train_pca,
        y_train,
        X_test_pca,
        y_test,
        pc1_label,
        pc2_label,
        "Scikit-learn logistic regression — Wine, 13 features\n"
        "Decision regions on the first two principal components",
        "wine_sklearn_pca_slice.png",
    )

    return {"Wine — 13 features": losses}


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    losses = {}
    losses.update(run_iris_experiments())
    losses.update(run_wine_experiment())
    plot_training_losses(losses)


if __name__ == "__main__":
    main()
