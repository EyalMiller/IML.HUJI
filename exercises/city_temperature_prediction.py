from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    x = pd.read_csv(filename, parse_dates=['Date'])
    y = x.pop('Temp')

    # There is no big city with temperature of -70... Especially not Capetown or Amman
    x = x.drop(x[y <= -70].index)
    y = y.drop(y[y <= -70].index)

    # I looked at the data and there is only one city from every country, so there is no point
    # to keep the cities too.
    x = x.drop(['City'], axis=1)

    # Create the DayOfYear column
    day_of_year = x['Date'].apply(lambda d: d.strftime('%j')).astype(int)
    x = pd.concat([x, day_of_year], axis=1)
    x.columns.values[5] = "DayOfYear"
    return x, y


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    x, y = load_data('City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    x_israel = x[x['Country'] == "Israel"]
    y_israel = y[x['Country'] == "Israel"]
    years = x_israel['Year'].unique()
    for year in years:
        plt.scatter(x_israel['DayOfYear'][x_israel['Year'] == year], y_israel[x_israel['Year'] == year], alpha=0.8)
    plt.legend(years, fontsize=8)
    plt.xlabel("Day of the year")
    plt.ylabel("Temperature (C)")
    plt.title("The temperature over a year in Israel")
    plt.show()

    israel_samples = pd.concat([x_israel, y_israel], axis=1)
    month_temp_std = israel_samples.groupby('Month').agg({"Temp": 'std'})
    months = israel_samples['Month'].unique()
    plt.bar(months, month_temp_std['Temp'], color='green')
    plt.xticks(months)
    plt.title("Standard Deviation of the average temperature in every month in Israel", fontdict={'fontsize': 10})
    plt.xlabel("The month")
    plt.ylabel("Standard deviation")
    plt.show()

    # Question 3 - Exploring differences between countries
    all_samples = pd.concat([x, y], axis=1)
    color_map = {"Israel": 'black', "Jordan": 'Yellow', "South Africa": 'blue', "The Netherlands": 'red'}
    grouped_samples = all_samples.groupby(['Country', 'Month'])
    for country in all_samples['Country'].unique():
        std = np.array([])
        mean = np.array([])
        for month in all_samples['Month'].unique():
            specific_sample = grouped_samples.get_group((country, month))
            country_temp_std = specific_sample.agg({'Temp': ['mean', 'std']})
            mean = np.append(mean, country_temp_std.iloc[0])
            std = np.append(std, country_temp_std.iloc[1])
        plt.errorbar(months, mean, yerr=std, label=country, elinewidth=1.3, color=color_map[country])
    plt.xticks(months)
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Temperature (C)")
    plt.title("Average temperature over the year in various countries")
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    x_israel = x_israel['DayOfYear']
    x_israel_train, y_israel_train, x_israel_test, y_israel_test = split_train_test(x_israel, y_israel, 0.75)
    losses = np.zeros(10)
    k_values = np.linspace(1, 10, 10).astype(int)
    for k in k_values:
        poly_model = PolynomialFitting(k)
        poly_model.fit(x_israel_train.to_numpy(), y_israel_train.to_numpy())
        loss = poly_model.loss(x_israel_test.to_numpy(), y_israel_test.to_numpy())
        print("Loss for k = " + str(k) + " is " + str(np.round(loss, 2)))
        losses[k - 1] = loss
    plt.bar(k_values, losses, color='blue')
    plt.xticks(k_values)
    plt.title("Model loss as a function of the k value - temperature in Israel")
    plt.xlabel("k")
    plt.ylabel("loss [MSE]")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    countries = ['South Africa', 'The Netherlands', 'Jordan']
    results = np.zeros(len(countries))
    israel_overfit_model = PolynomialFitting(k=3)
    israel_overfit_model.fit(x_israel.to_numpy(), y_israel.to_numpy())
    for i in range(len(countries)):
        x_country_test = x[x['Country'] == countries[i]]['DayOfYear']
        y_country_test = y[x['Country'] == countries[i]]
        loss = israel_overfit_model.loss(x_country_test.to_numpy(), y_country_test.to_numpy())
        results[i] = loss
    plt.bar(countries, results, color='maroon', width=0.5)
    plt.title("Model loss in different countries, when fitted on Israel")
    plt.xlabel("The Country")
    plt.ylabel("loss [MSE]")
    plt.show()
