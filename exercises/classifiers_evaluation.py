from math import atan2

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


NUM_OF_TRIES = 1000
losses = None


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    global losses
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f)
        x = data[:, :2]
        y = data[:, 2]

        # Fit Perceptron and record loss in each fit iteration
        def my_callback(fit: Perceptron, failed_x: np.ndarray, failed_y: int):
            global losses
            loss = fit.loss(x, y)
            losses = np.append(losses, loss)

        losses = np.array([])
        Perceptron(callback=my_callback).fit(x, y)

        # Plot figure
        plt.plot(losses, '-o', alpha=0.5)
        plt.title("Train Loss as a function of the number of changes - " + n + " Module:", fontdict={'fontsize': 10})
        plt.xlabel("# of changes")
        plt.ylabel("misclassification loss")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return mu[0] + xs, mu[1] + ys


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(f)
        x = data[:, :2]
        y = data[:, 2]

        lda = LDA()
        bayes = GaussianNaiveBayes()

        # Fit models and predict over training set
        lda.fit(x, y)
        bayes.fit(x, y)

        pred_lda = lda.predict(x)
        pred_bayes = bayes.predict(x)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        accuracy_lda = accuracy(y, pred_lda)
        accuracy_bayes = accuracy(y, pred_bayes)
        fig, (lda_axis, bayes_axis) = plt.subplots(1, 2, figsize=(9, 3))
        lda_axis.set_title("LDA Model, Accuracy: " + str(accuracy_lda))
        bayes_axis.set_title("Bayes Model, Accuracy: " + str(accuracy_bayes))
        shape_options = ['o', 's', 'd']
        color_options = ['r', 'g', 'b']
        lda_mu = lda.mu_
        bayes_mu = bayes.mu_
        for i in range(3):
            for j in range(3):
                x_to_plot_1_lda = x[:, 0][np.where((y == i) & (pred_lda == j))]
                x_to_plot_2_lda = x[:, 1][np.where((y == i) & (pred_lda == j))]
                x_to_plot_1_bayes = x[:, 0][np.where((y == i) & (pred_bayes == j))]
                x_to_plot_2_bayes = x[:, 1][np.where((y == i) & (pred_bayes == j))]
                marker = shape_options[i]
                color = color_options[j]
                lda_axis.scatter(x_to_plot_1_lda, x_to_plot_2_lda, marker=marker, c=color, alpha=0.7)
                bayes_axis.scatter(x_to_plot_1_bayes, x_to_plot_2_bayes, marker=marker, c=color, alpha=0.7)
            ellipse = get_ellipse(lda_mu[i], lda.cov_)
            lda_axis.plot(ellipse[0], ellipse[1], c='black')
            ellipse = get_ellipse(bayes_mu[i], np.diag(bayes.vars_[i]))
            bayes_axis.plot(ellipse[0], ellipse[1], c='black')
        # print(bayes.vars_)
        lda_axis.scatter(lda_mu[:, 0], lda_mu[:, 1], marker='x', color='black')
        bayes_axis.scatter(bayes_mu[:, 0], bayes_mu[:, 1], marker='x', color='black')
        fig.suptitle("dataset plot - " + f[:-4], fontsize=16)
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
