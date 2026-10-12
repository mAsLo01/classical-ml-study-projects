import numpy as np


def softmax(logits):
    """Computes numerically stable softmax probabilities."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def mylearn(X, y, learning_rate=0.1, num_iterations=5000):
    """
    Trains multiclass logistic regression with gradient descent.

    Class labels must be consecutive integers starting from zero.
    Returns the weight matrix and the cross-entropy history.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    classes = np.unique(y)
    expected_classes = np.arange(classes.size)
    if not np.array_equal(classes, expected_classes):
        raise ValueError(
            "Class labels must be consecutive integers starting from zero."
        )

    n_objects, n_features = X.shape
    n_classes = classes.size

    X_bias = np.column_stack((np.ones(n_objects), X))
    weights = np.zeros((n_classes, n_features + 1))
    y_one_hot = np.eye(n_classes)[y]
    losses = []

    for _ in range(num_iterations):
        logits = X_bias @ weights.T
        probabilities = softmax(logits)

        gradient = (probabilities - y_one_hot).T @ X_bias / n_objects
        weights -= learning_rate * gradient

        clipped_probabilities = np.clip(probabilities, 1e-15, 1.0)
        loss = -np.mean(
            np.sum(y_one_hot * np.log(clipped_probabilities), axis=1)
        )
        losses.append(loss)

    return weights, np.asarray(losses)


def mypredict_proba(weights, X):
    """Returns class probabilities for each observation."""
    X = np.asarray(X, dtype=float)
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    logits = X_bias @ weights.T
    return softmax(logits)


def mypredict(weights, X):
    """Returns the class with the highest predicted probability."""
    probabilities = mypredict_proba(weights, X)
    return np.argmax(probabilities, axis=1)
