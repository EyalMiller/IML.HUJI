from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.sort(np.unique(y))
        num_of_classes = len(self.classes_)
        self.pi_ = np.zeros(num_of_classes)
        self.mu_ = np.zeros((num_of_classes, x.shape[1]))
        self.cov_ = np.zeros((x.shape[1], x.shape[1]))
        for i in range(len(self.classes_)):
            indices = np.where(y == self.classes_[i], np.ones(len(y)), np.zeros(len(y)))
            self.pi_[i] = np.sum(indices) / len(y)
            self.mu_[i] = np.sum(x[indices == 1], 0) / np.sum(indices)
        for i in range(x.shape[0]):
            mu = self.mu_[np.where(self.classes_ == y[i])]
            self.cov_ += np.dot(np.transpose(x[i] - mu), x[i] - mu)
        self.cov_ /= (x.shape[0] - num_of_classes)
        self._cov_inv = inv(self.cov_)

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
        results = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            k_predict = np.zeros(len(self.classes_))
            for j in range(len(self.classes_)):
                k_predict[j] = np.dot(np.transpose(np.dot(self._cov_inv, self.mu_[j])), x[i]) + np.log(
                    self.pi_[j]) - np.dot(np.dot(self.mu_[j], self._cov_inv), self.mu_[j]) / 2
            results[i] = self.classes_[np.argmax(k_predict)]
        return results

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood = np.zeros((x.shape[0], len(self.classes_)))
        for i in range(x.shape[0]):
            for j in range(len(self.classes_)):
                exp_arg = - np.dot(np.dot(np.transpose(x[i] - self.mu_[j]), self._cov_inv), x[i] - self.mu_[j]) / 2
                likelihood[i][j] = 1 / (np.sqrt(np.power(2 * np.pi, len(self.classes_)) * det(self.cov_))) * np.exp(
                    exp_arg) * self.pi_[j]
        return likelihood

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self._predict(x))
