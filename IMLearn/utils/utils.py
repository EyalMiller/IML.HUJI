from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(x: pd.DataFrame, y: pd.Series, train_proportion: float = .25) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    x : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    indices = np.linspace(0, x.shape[0] - 1, x.shape[0]).astype(int)
    chosen_train_indices = np.random.choice(indices, np.ceil(train_proportion * x.shape[0]).astype(int))
    chosen_test_indices = np.delete(indices, chosen_train_indices)
    x_train = x.iloc[chosen_train_indices]
    x_test = x.iloc[chosen_test_indices]
    y_train = y.iloc[chosen_train_indices]
    y_test = y.iloc[chosen_test_indices]
    return x_train, y_train, x_test, y_test


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
