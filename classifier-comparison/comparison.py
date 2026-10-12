import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import label_binarize


def train_models(models, X_train, X_test, y_train, y_test, classes, cv):
    """
    Compares models with cross-validation on the training set and then
    evaluates fitted models once on the held-out test set.
    """
    results = {}
    predictions = {}
    probabilities = {}

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    for name, model in models.items():
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        predictions[name] = y_pred
        probabilities[name] = y_proba

        results[name] = {
            "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
            "CV Accuracy Std": cv_scores["test_accuracy"].std(),
            "CV Macro F1 Mean": cv_scores["test_macro_f1"].mean(),
            "CV Macro F1 Std": cv_scores["test_macro_f1"].std(),
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision (macro)": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "Test Recall (macro)": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "Test F1 (macro)": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "Test ROC AUC (macro OvR)": roc_auc_score(
                y_test,
                y_proba,
                labels=classes,
                multi_class="ovr",
                average="macro",
            ),
        }

    return results, predictions, probabilities


def plot_roc(probabilities, y_test, classes, output_path):
    """Plots macro-averaged one-vs-rest ROC curves."""
    y_test_binary = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_proba in probabilities.items():
        false_positive_rates = []
        true_positive_rates = []
        class_auc_values = []

        for class_index in range(n_classes):
            fpr, tpr, _ = roc_curve(
                y_test_binary[:, class_index],
                y_proba[:, class_index],
            )
            false_positive_rates.append(fpr)
            true_positive_rates.append(tpr)
            class_auc_values.append(auc(fpr, tpr))

        common_fpr = np.unique(np.concatenate(false_positive_rates))
        mean_tpr = np.zeros_like(common_fpr)

        for fpr, tpr in zip(false_positive_rates, true_positive_rates):
            mean_tpr += np.interp(common_fpr, fpr, tpr)

        mean_tpr /= n_classes
        macro_auc = float(np.mean(class_auc_values))

        ax.plot(
            common_fpr,
            mean_tpr,
            linewidth=2,
            label=f"{name} (macro AUC = {macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Macro-averaged one-vs-rest ROC curves")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    predictions,
    y_test,
    classes,
    class_names,
    output_path,
):
    """Plots one confusion matrix for each classifier."""
    n_models = len(predictions)
    n_columns = 2
    n_rows = int(np.ceil(n_models / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 5 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_test, y_pred, labels=classes)
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_names,
        )
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Confusion matrices on the test set", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(results, baseline_accuracy, output_path):
    """Plots held-out test accuracy and macro F1 for each model."""
    names = list(results)
    accuracies = [results[name]["Test Accuracy"] for name in names]
    macro_f1 = [results[name]["Test F1 (macro)"] for name in names]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_bars = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Test accuracy",
    )
    f1_bars = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Test macro F1",
    )

    ax.axhline(
        baseline_accuracy,
        color="black",
        linestyle="--",
        label=f"Majority-class baseline ({baseline_accuracy:.3f})",
    )
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)

    ax.set_xticks(x, names)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classifier performance on the held-out test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
