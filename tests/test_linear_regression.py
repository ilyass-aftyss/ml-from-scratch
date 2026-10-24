
import numpy as np
import pytest

from src.linear_regression import LinearRegression


def test_model_learns_linear_relationship():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 3 * X + 5

    model = LinearRegression(
        learning_rate=0.01,
        epochs=3000
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert np.mean((predictions - y) ** 2) < 0.01


def test_predict_before_fit_raises_error():
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_mismatched_sample_counts_raise_error():
    model = LinearRegression()

    X = np.array([[1.0], [2.0]])
    y = np.array([[3.0]])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_learning_rate_raises_error():
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=0)


def test_nan_input_raises_error():
    model = LinearRegression()

    X = np.array([[1.0], [np.nan]])
    y = np.array([[3.0], [5.0]])

    with pytest.raises(ValueError):
        model.fit(X, y)
