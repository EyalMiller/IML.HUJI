from __future__ import annotations
from typing import NoReturn
from IMLearn.base.base_estimator import BaseEstimator
from IMLearn.learners.regressors.linear_regression import LinearRegression
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.estimator = LinearRegression(False)
        self.k = k

    def _fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.estimator.fit(self.__transform(x), y)

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
        return self.estimator.predict(self.__transform(x))


    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
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
        return self.estimator.loss(self.__transform(x), y)


    def __transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        x: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(x, self.k + 1, increasing=True)
