from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    eps = np.random.normal(0, np.sqrt(5), n_samples)
    y_true = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = y_true + eps
    plt.scatter(X, y_true)
    plt.title("True model - noiseless")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), 0.66)
    x_train = x_train.to_numpy().flatten()
    y_train = y_train.to_numpy().flatten()
    x_test = x_test.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()
    plt.scatter(x_train, y_train)
    plt.title("Train model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.scatter(x_test, y_test)
    plt.title("Test model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = np.zeros(11)
    test_errors = np.zeros(11)
    for k in range(11):
        poly_model = PolynomialFitting(k)
        train_errors[k], test_errors[k] = cross_validate(poly_model, x_train, y_train,
                                                         mean_square_error)
    plt.plot(range(11), train_errors, test_errors)
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.title("Polynomial fitting error as a function of k")
    plt.legend(["train", "validation"])
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
