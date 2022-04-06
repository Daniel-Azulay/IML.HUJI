import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os.path
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
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
    data = pd.read_csv(filename, parse_dates=['Date'])
    data = data.dropna()
    data = data.drop(data.index[(data.Day < 0) | (data.Month < 0) | (data.Year < 0) | (data.Day > 31) |
                                (data.Month > 12) | (data.Temp < -20)])
    temp_col = data.pop('Temp')
    data['DayOfYear'] = data.Date.apply(lambda x: x.dayofyear)
    data['Temp'] = temp_col

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    path = '..\datasets\City_Temperature.csv'
    samples = load_data(os.path.join(os.path.dirname(__file__), path))

    # Question 2 - Exploring data for specific country
    israeli_samples = samples.loc[samples['Country'] == 'Israel']
    fig = px.scatter(israeli_samples, x='DayOfYear', y='Temp', color=israeli_samples['Year'].astype(str),
                     height=600, width=800).show()
    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()