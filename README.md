# Classical Machine Learning Study Projects

Учебные проекты по реализации и исследованию методов классического
машинного обучения.

## Projects

### 1. Binary logistic regression

Бинарная классификация Iris: собственная реализация логистической
регрессии методом градиентного спуска и сравнение со scikit-learn.

### 2. Multiclass softmax regression

Собственная реализация многоклассовой softmax-регрессии. Эксперименты
на датасетах Iris и Wine, стандартизация данных и PCA-визуализация.

### 3. Classifier comparison

Сравнение SVM, Random Forest, Gaussian Naive Bayes и XGBoost на Iris.
Модели оцениваются с помощью кросс-валидации и отложенной тестовой выборки.

### 4. Regression model comparison

Сравнение линейной регрессии, Ridge, Lasso, дерева решений и случайного
леса на Diabetes. Гиперпараметры подбираются кросс-валидацией без
использования тестовой выборки.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt