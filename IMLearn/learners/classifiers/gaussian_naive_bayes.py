from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from scipy.linalg import det, inv
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((num_of_classes, x.shape[1]))
        for i in range(len(self.classes_)):
            indices = np.where(y == self.classes_[i], np.ones(len(y)), np.zeros(len(y)))
            self.pi_[i] = np.sum(indices) / len(y)
            self.mu_[i] = np.sum(x[indices == 1], 0) / np.sum(indices)
            for sample in x[indices == 1]:
                for j in range(x.shape[1]):
                    self.vars_[i][j] += (sample[j] - self.mu_[i][j]) * (sample[j] - self.mu_[i][j])
            self.vars_[i] /= (np.sum(indices) - 1)


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
        # create an 'artificial' inverse covariance matrix for every k:
        results = np.zeros(x.shape[0])
        likelihood = self.likelihood(x)
        for i in range(x.shape[0]):
            results[i] = self.classes_[np.argmax(likelihood[i])]
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
        cov = np.zeros((len(self.classes_), x.shape[1], x.shape[1]))
        inv_cov = np.zeros((len(self.classes_), x.shape[1], x.shape[1]))
        for j in range(len(self.classes_)):
            cov[j] = np.diag(self.vars_[j])
            inv_cov[j] = inv(cov[j])
        for i in range(x.shape[0]):
            for j in range(len(self.classes_)):
                exp_arg = - np.dot(np.dot(np.transpose(x[i] - self.mu_[j]), inv_cov[j]), x[i] - self.mu_[j]) / 2
                likelihood[i][j] = 1 / (np.sqrt(np.power(2 * np.pi, len(self.classes_)) * det(cov[j]))) * np.exp(
                    exp_arg) * self.pi_[j]
        # print(likelihood)
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
