

import numpy as np

from src.metrics import mse, mae, r2_score


def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    assert mse(y_true, y_pred) == 0


def test_mae():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 4])

    assert mae(y_true, y_pred) == 2 / 3


def test_r2():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    assert r2_score(y_true, y_pred) == 1