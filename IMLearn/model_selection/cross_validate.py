from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    d = 1
    if len(X.shape) > 1:
        d = X.shape[1]
    train_score = np.zeros(cv)
    test_score = np.zeros(cv)
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    for i in range(cv):
        X_val = X_folds[i]
        y_val = y_folds[i]
        if d > 1:
            X_train = np.zeros((len(X) - len(X_val), d))
        else:
            X_train = np.zeros(len(X) - len(X_val))
        y_train = np.zeros(len(X) - len(X_val))
        ind = 0
        for j in range(cv):
            if i != j:
                for k in range(len(X_folds[j])):
                    X_train[ind] = X_folds[j][k]
                    y_train[ind] = y_folds[j][k]
                    ind += 1
        estimator.fit(X_train, y_train)
        y_pred_train = estimator.predict(X_train)
        y_pred_val = estimator.predict(X_val)
        train_score[i] = scoring(y_train, y_pred_train)
        test_score[i] = scoring(y_val, y_pred_val)
    return np.mean(train_score), np.mean(test_score)
