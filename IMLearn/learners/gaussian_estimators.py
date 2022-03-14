from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
import matplotlib.pyplot as plt


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False):
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ - bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_ - float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_ - float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.x_values = None
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, x: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        x: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = x.mean()
        self.var_ = x.var()
        self.fitted_ = True
        return self

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        x: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf = 1 / np.sqrt(2 * np.pi * self.var_) * np.exp(-1 * np.power(x - self.mu_, 2) / (2 * self.var_))
        return pdf

    @staticmethod
    def log_likelihood(mu: float, sigma: float, x: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        x : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        sum_helper = 0
        for sample in x:
            sum_helper += (sample - mu) * (sample - mu)
        log_likelihood = np.log(1 / np.power(2 * np.pi * sigma * sigma, len(x) / 2)) * np.exp(
            -1 / (2 * sigma * sigma) * sum_helper)
        return log_likelihood


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ - bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_ - ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_ - ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, x: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(x, axis=0)
        self.cov_ = np.cov(x, rowvar=False)
        # print(self.mu_)
        # print(self.cov_)
        self.fitted_ = True
        return self

    def pdf(self, x: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        my_pdf = np.array([])
        for sample in x:
            my_pdf = np.append(my_pdf, 1 / np.sqrt(np.power(2 * np.pi, len(x[0])) * np.exp(det(self.cov_)))
                               * np.exp(
                -1 / 2 * np.matmul((sample - self.mu_), np.matmul(inv(self.cov_), np.transpose(sample - self.mu_)))))
        return my_pdf

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        x : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        m = len(x)
        # the trace is here because I wanted the sum over the same X samples, which means that I need only the values
        # in the main diagonal
        sum_helper = np.matmul((x - mu), np.matmul(inv(cov), np.transpose(x - mu))).trace()
        log_likelihood = -len(x[0]) * m / 2 * np.log(2 * np.pi) - m / 2 * slogdet(cov)[1] - 1 / 2 * sum_helper
        return log_likelihood
