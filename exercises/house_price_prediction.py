import os.path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn import utils

pio.templates.default = "simple_white"


def load_data(filename: str):
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
    data = pd.read_csv(filename)
    data = data.dropna()
    # converting dates to age
    data["date"] = data["date"].apply(parse_date)
    data = data.drop(data.index[data.date == ""])
    data["date"] = pd.to_datetime(data["date"])
    data["year_of_purchase"] = data.date.dt.year
    data = data.drop(data.index[data.year_of_purchase <= 0])

    data["age_of_house"] = data["year_of_purchase"] - data["yr_built"]
    data["time_since_renovation"] = data["age_of_house"]
    data.time_since_renovation.where(data.yr_renovated == 0, data.year_of_purchase - data.yr_renovated)

    # delete negative prices, negative age of house\time since renovation, negative bedrooms\bathrooms\sqfts
    data = data.drop(data.index[(data.price < 0) | (data.age_of_house < 0) | (data.bedrooms < 0) | (data.bathrooms < 0)
                     | (data.sqft_living <= 0) | (data.sqft_above < 0) | (data.sqft_basement < 0) |
                    (data.sqft_living15 < 0) | (data.sqft_lot15 < 0) | (data.time_since_renovation < 0)])


    # make id as index, get dummies for zipcode
    data = pd.get_dummies(data, columns=["zipcode"])
    # split samples and response, and remove unnesessary columns
    y_response = data["price"]
    remove_list = ["price", "date", "yr_built", "year_of_purchase", "id", "lat", "long", "sqft_living15",
                   "sqft_lot15"]
    for col in remove_list:
        data = data.drop(col, axis=1)
    return data, y_response


def parse_date(date: str) -> str:
    if len(date) < 8:
        return ""
    return date[:8]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_of_y = np.std(y)
    for col in X.columns:
        pearson_cor = np.cov(X[col].to_numpy(), y) / (np.std(X[col].to_numpy()) * std_of_y)
        graph_title = "Price as a function of " + col + ". Pearson correlation is: " + str(pearson_cor[0][1])
        fig = go.Figure([go.Scatter(x=np.array(X[col]), y=y, mode="markers", marker=dict(color="black"))],
                        layout=go.Layout(title=graph_title,
                                         xaxis={"title": col},
                                         yaxis={"title": "Price"},
                                         height=400))
        cur_output_path = os.path.join(os.path.dirname(__file__), output_path) + "\\" + col + ".jpg"
        fig.write_image(cur_output_path)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = '..\datasets\house_prices.csv'
    samples, response = load_data(os.path.join(os.path.dirname(__file__), path))
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(samples, response)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = utils.split_train_test(samples, response, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        lin_model = LinearRegression()
        p_mse_arr = []
        for i in range(10):
            i_train_x, i_train_y = utils.split_train_test(train_x, train_y, train_proportion=p / 100)[:2]
            lin_model.fit(i_train_x, i_train_y)
            p_mse_arr.append(lin_model.loss(test_x, test_y))
        mean_loss.append(np.mean(p_mse_arr))
        std_loss.append(np.std(p_mse_arr))
    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)
    # mean_loss_fig = go.Figure([go.Scatter(x=np.arange(10, 101), y=mean_loss, mode="markers", marker=dict(color="black"))],
    #                 layout=go.Layout(title="mean of loss over 10 iterations, as function of training size",
    #                                  xaxis={"title": "percentage of the training set used for training"},
    #                                  yaxis={"title": "mean of loss"},
    #                                  height=400))
    # mean_loss_fig.show()
    go.Figure([go.Scatter(x=np.arange(10, 101), y=mean_loss - 2 * std_loss, fill=None, mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=np.arange(10, 101), y=mean_loss + 2 * std_loss, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=np.arange(10, 101), y=mean_loss, mode="markers+lines", marker=dict(color="black", size=1),
                          showlegend=False)],
              layout=go.Layout(title="mean of loss over 10 iterations, as function of training size",
                                     xaxis={"title": "percentage of the training set used for training"},
                                     yaxis={"title": "mean of loss"},
                                     height=400)).show()
