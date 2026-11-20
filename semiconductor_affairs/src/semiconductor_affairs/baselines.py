"""Baseline models. All take X of shape (n, window, n_features)."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class Zero:
    """Always predict 0: 'the next hour is unpredictable'."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Persistence:
    """Predict that the target's last return repeats."""

    def __init__(self, target_col: int):
        self.target_col = target_col

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, -1, self.target_col]


def flatten(X):
    """(n, window, features) -> (n, window*features) for linear models."""
    return X.reshape(len(X), -1)


def ridge():
    """Linear regression on all lagged features, with an L2 penalty.

    The penalty strength (alpha) is chosen by time-ordered CV on the
    training data only. Scaling is fit on training data only too.
    """
    return make_pipeline(
        FunctionTransformer(flatten),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(0, 6, 25), cv=TimeSeriesSplit(5)),
    )
