from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import matplotlib.pyplot as plt


from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
'''import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"'''


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # load data
    x = pd.read_csv(filename)
    y = x.pop('price')

    # price should be more than 0
    x = x.drop(x[y <= 0].index)
    y = y.drop(y[y <= 0].index)

    # ID - total irrelevant data in my opinion, basically just like an index
    x = x.drop(['id'], axis=1)

    # there should be at least one bedroom
    y = y.drop(y[x['bedrooms'] <= 0].index)
    y = y.drop(y[x['bedrooms'] > 20].index)
    x = x.drop(x[x['bedrooms'] <= 0].index)
    x = x.drop(x[x['bedrooms'] > 20].index)

    # bathroom is necessary, so there should be at least one
    y = y.drop(y[x['bathrooms'] <= 0].index)
    x = x.drop(x[x['bathrooms'] <= 0].index)

    # make zipcode a dummy feature, and also date
    x['zipcode'] = x['zipcode'].astype('category')
    x['zipcode'] = x['zipcode'].cat.reorder_categories(x['zipcode'].unique(), ordered=True)
    x['zipcode'] = x['zipcode'].cat.codes

    x['date'] = x['date'].astype('category')
    x['date'] = x['date'].cat.codes

    # drop nan values
    x = x.drop(y.index[y.apply(np.isnan)])
    y = y.drop(y.index[y.apply(np.isnan)])

    # Some more ideas: add a column for "has a basement", add a column for "was renovated".
    x['was_renovated'] = x['yr_renovated'].where(x['yr_renovated'] == 0, 1).astype(int)
    x['has_basement'] = x['sqft_basement'].where(x['sqft_basement'] == 0, 1).astype(int)

    return x, y


def feature_evaluation(x: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    x : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for i in range(x.shape[1]):
        feature = x.iloc[:, i]
        feature_name = str(x.columns[i])
        pearson_cor = y.cov(feature) / (y.std() * feature.std())
        plt.scatter(feature, y, color=['red'], alpha=0.4)
        plt.title("Correlation for feature: " + feature_name + ", r = " + str(pearson_cor), fontdict={'fontsize': 10})
        plt.xlabel(feature_name)
        plt.ylabel("price")
        plt.savefig(output_path + "\\" + feature_name + ".jpeg")
        plt.clf()


def choose_part_of_data(x_train: pd.DataFrame, y_train: pd.Series, train_proportion) -> Tuple[pd.DataFrame, pd.Series]:
    indices = np.linspace(0, x_train.shape[0] - 1, x_train.shape[0]).astype(int)
    chosen_indices = np.random.choice(indices, np.ceil(train_proportion * x_train.shape[0]).astype(int))
    x_chosen = x_train.iloc[chosen_indices]
    y_chosen = y_train.iloc[chosen_indices]
    return x_chosen, y_chosen


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    # I transferred the csv to the same directory because pycharm couldn't recognize other directories...
    x, y = load_data(r'house_prices.csv')
    y = y.squeeze()

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y.squeeze(), "graphs")

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(x, y, 0.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    losses = np.zeros(91)
    variances = np.zeros(91)
    percent_arr = np.linspace(10, 100, 91)
    for percent in percent_arr:
        loss = np.zeros(10)
        for i in range(10):
            x_chosen, y_chosen = choose_part_of_data(x_train, y_train, percent / 100)
            linear_estimator = LinearRegression()
            linear_estimator.fit(x_chosen.to_numpy(), y_chosen.to_numpy())
            loss[i] = linear_estimator.loss(x_test.to_numpy(), y_test.to_numpy())
        losses[int(percent - 10)] = loss.mean()
        variances[int(percent - 10)] = loss.std()
    plt.plot(percent_arr, losses, 'c')
    plt.errorbar(percent_arr, losses, yerr=variances, ecolor='red', elinewidth=0.5)
    plt.title("Mean loss and Standard Deviation of test as a function of data size", fontdict={'fontsize': 10})
    plt.xlabel("percentage of chosen train data out of all data")
    plt.ylabel("mean of loss")
    plt.show()
