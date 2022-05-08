from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np


def weighted_loss(y_true, y_pred):
    return np.sum(np.where(np.sign(y_pred) != np.sign(y_true), np.abs(y_true), np.zeros(y_true.shape[0])))


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the jth feature is about the threshold
    """
    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        opt_thr = 0
        min_thr_err = 1
        opt_thr_feature = 0
        for i in range(x.shape[1]):
            for sign in [-1, 1]:
                thr, thr_err = self._find_threshold(x[:, i], y, sign)
                if thr_err <= min_thr_err:
                    self.sign_ = sign
                    min_thr_err = thr_err
                    opt_thr = thr
                    opt_thr_feature = i
        self.threshold_ = opt_thr
        self.j_ = opt_thr_feature

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

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        pred = np.where(x[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)
        return pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_values = np.sort(values)
        # To also have the possible option that everything could be zero:
        sorted_values = np.append(sorted_values, sorted_values[-1] + 1)
        thr = 0
        thr_err = 1
        count = 0
        for value in sorted_values:
            count += 1
            y_th = np.where(values >= value, sign, -1 * sign)
            err = weighted_loss(labels, y_th)
            if err <= thr_err:
                thr = value
                thr_err = err
        return thr, thr_err

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
        y_pred = self._predict(x)
        return weighted_loss(y, y_pred)
