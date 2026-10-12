import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def mylearn(X, y, learning_rate=0.1, num_iterations=5000):
    n_objects, n_features = X.shape
    X_bias = np.column_stack((np.ones(n_objects), X))
    theta = np.zeros(n_features + 1)
    losses = []

    for _ in range(num_iterations):
        probabilities = sigmoid(X_bias @ theta)
        gradient = X_bias.T @ (probabilities - y) / n_objects
        theta -= learning_rate * gradient

        clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
        )
        losses.append(loss)

    return theta, np.asarray(losses)


def mypredict_proba(theta, X):
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    return sigmoid(X_bias @ theta)


def mypredict(theta, X, threshold=0.5):
    probabilities = mypredict_proba(theta, X)
    return (probabilities >= threshold).astype(np.int32)
