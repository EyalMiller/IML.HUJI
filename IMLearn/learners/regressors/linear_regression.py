from __future__ import annotations
from typing import NoReturn
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_, self.coefficients = include_intercept, None

    def _fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            ones = np.full(x.shape[0], 1)
            x = np.insert(x, 0, ones, 1)
        self.coefficients = np.matmul(pinv(x), y)


    def _predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            ones = np.full(x.shape[0], 1)
            x = np.insert(x, 0, ones, 1)
        return np.matmul(x, self.coefficients)

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self.predict(x))
