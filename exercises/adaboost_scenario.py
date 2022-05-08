import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    booster = AdaBoost(DecisionStump, n_learners)
    booster.fit(train_X, train_y)
    train_losses = [booster.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    test_losses = [booster.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
    x = np.arange(n_learners) + 1
    plt.plot(x, train_losses)
    plt.plot(x, test_losses)
    plt.legend(['train', 'test'])
    plt.title("train and test loss as a function of the number of models")
    plt.xlabel("number of weak learners")
    plt.ylabel("loss of adaboost")
    plt.show()

    # Question 2: Plotting decision surfaces
    # Fine, I give up - you finally made me use plotly...

    T = [5, 50, 100, 250]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{Number of iterations: {t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Models Dataset:}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    symbols = np.array(["x", "circle", "x"])
    lims = np.array([test_X.min(axis=0), test_X.max(axis=0)]).T + np.array([-.4, .4])
    for i, iteration in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda some_x: booster.partial_predict(some_x, iteration), lims[0], lims[1],
                              showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                    colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    opt_size = np.argmin(test_losses) + 1
    lowest_loss = np.amin(test_losses)
    print("Number of weak learners which reached min test loss:", opt_size)
    print("It reached a loss of", lowest_loss)

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title=rf"$\textbf{{Optimal size - {opt_size}, loss - {lowest_loss}}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.add_traces(
        [decision_surface(lambda some_x: booster.partial_predict(some_x, opt_size), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))], rows=1, cols=1)
    fig.show()
    # Question 4: Decision surface with weighted samples
    weights = booster.D_[-1]
    weights = weights / np.amax(weights) * 10
    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title=rf"$\textbf{{Training set, with size according to weights}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.add_traces(
        [decision_surface(lambda some_x: booster.partial_predict(some_x, n_learners), lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1), size=weights))], rows=1, cols=1)
    fig.show()


if __name__ == '__main__':
    for error in [0, 0.4]:
        np.random.seed(0)
        fit_and_evaluate_adaboost(error)
