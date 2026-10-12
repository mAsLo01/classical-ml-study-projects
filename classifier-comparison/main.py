from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from comparison import (
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_roc,
    train_models,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


RANDOM_STATE = 42
TEST_SIZE = 0.3
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def build_models(n_classes):
    """Creates the classifiers used in the comparison."""
    models = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(kernel="linear"),
                        method="sigmoid",
                        cv=5,
                        ensemble=False,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "Gaussian NB": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
    else:
        print(
            "Warning: XGBoost is not installed and will be skipped. "
            "Install it with: python -m pip install xgboost"
        )

    return models


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target
    classes = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(n_classes=len(classes))
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results, predictions, probabilities = train_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classes=classes,
        cv=cv,
    )

    majority_class = np.bincount(y_train).argmax()
    majority_baseline = np.mean(y_test == majority_class)

    plot_roc(
        probabilities,
        y_test,
        classes,
        FIGURES_DIR / "roc_curves.png",
    )
    plot_confusion_matrices(
        predictions,
        y_test,
        classes,
        iris.target_names,
        FIGURES_DIR / "confusion_matrices.png",
    )
    plot_metric_comparison(
        results,
        majority_baseline,
        FIGURES_DIR / "metric_comparison.png",
    )

    results_table = pd.DataFrame(results).T.round(4)
    results_table = results_table.sort_values(
        "CV Macro F1 Mean",
        ascending=False,
    )

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(results_table.to_string())

    best_model_name = results_table["CV Macro F1 Mean"].idxmax()
    best_cv_score = results_table.loc[best_model_name, "CV Macro F1 Mean"]
    best_test_score = results_table.loc[best_model_name, "Test F1 (macro)"]

    print(
        "\nBest model by training-set cross-validation: "
        f"{best_model_name} "
        f"(CV macro F1 = {best_cv_score:.4f}; "
        f"test macro F1 = {best_test_score:.4f})"
    )


if __name__ == "__main__":
    main()
