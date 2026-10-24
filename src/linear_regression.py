
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate doit être positif.")

        if epochs < 1 or not isinstance(epochs, int):
            raise ValueError("epochs doit être un entier positif.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X ne doit pas être vide.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contient des valeurs non finies.")

        return X

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y doit contenir une seule colonne.")

        if len(X) != len(y):
            raise ValueError("X et y doivent avoir le même nombre de lignes.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contient des valeurs non finies.")

        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y

            loss = np.mean(errors ** 2)

            if not np.isfinite(loss):
                raise FloatingPointError(
                    "Perte non finie : réduis le learning rate."
                )

            self.loss_history.append(float(loss))

            dw = (2 / n_samples) * (X.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if not (
                np.all(np.isfinite(self.weights))
                and np.isfinite(self.bias)
            ):
                raise FloatingPointError(
                    "Paramètres non finis : réduis le learning rate."
                )

        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Le modèle doit être entraîné avant predict().")

        X = self._validate_X(X)

        if X.shape[1] != len(self.weights):
            raise ValueError("Nombre de caractéristiques incorrect.")

        return X @ self.weights + self.bias
