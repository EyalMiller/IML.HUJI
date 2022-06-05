from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from numpy.linalg import pinv

from IMLearn.metrics import mean_square_error


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True):
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        d = 1
        if len(X.shape) > 1:
            d = X.shape[1]
        i_d = np.identity(d)
        zeros = np.zeros(d)
        if self.include_intercept_:
            ones = np.full(X.shape[0], 1)
            if d > 1:
                X = np.insert(X, 0, ones, 1)
            else:
                X = np.transpose(np.array([ones, X]))
            zeros = np.append(zeros, 0)
            i_d = np.identity(d + 1)
            i_d[0, 0] = 0
        if d > 1 or self.include_intercept_:
            X_lambda = np.append(X, np.sqrt(self.lam_) * i_d, axis=0)
        else:
            X_lambda = np.append(X, np.sqrt(self.lam_) * i_d)
        y_lambda = np.append(y, zeros)
        self.coefs_ = np.dot(pinv(X_lambda), y_lambda)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        d = 1
        if len(X.shape) > 1:
            d = X.shape[1]
        if self.include_intercept_:
            ones = np.full(X.shape[0], 1)
            if d > 1:
                X = np.insert(X, 0, ones, 1)
            else:
                X = np.transpose(np.array([ones, X]))
        return np.dot(X, self.coefs_)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))
