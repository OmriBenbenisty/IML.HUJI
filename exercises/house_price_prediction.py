from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

DROP = ["id", "price", "date", "yr_renovated", "zipcode", "lat", "long"]
KEEP = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "sqft_living15", "sqft_lot15"]

NUM_OF_BINS = 25  # should be a power of a number

TOTAL_AVERAGE = 0
AREA_BINS = np.zeros(NUM_OF_BINS)

"""
date -> year.month
renovated -> time since renovated : today.year - max(yr_built, yr_renovated)

long-lat -> average in section
            if new sample arrives and doesnt have bin, it gets the total average

"""


def get_area_bin(long: float, lat: float) -> int:
    return 0

def load_data(filename: str) -> (np.ndarray, np.ndarray):
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
    full_data = pd.read_csv(filename)

    # Remove prices under or equal to 0
    full_data = full_data.drop(full_data[full_data.price <= 0].index)

    data = full_data.drop(DROP, axis=1)

    # for col in data.head():
    #     print(f"\"{col}\", ", end="")
    # print()

    # Convert date to year.month 201412 -> 2014.12
    data["date"] = full_data["date"].apply(
        lambda x: pd.Timestamp(x).year + (pd.Timestamp(x).month / 10)
    )

    # add time since renovated : today.year - max(yr_built, yr_renovated)
    data["since_renovated"] = pd.DataFrame(
        full_data[["yr_built", "yr_renovated"]].max(axis=1)
        - pd.Timestamp.today().year).abs()


    # long lat to areas
    longitude_bins = np.linspace(full_data["long"].min(), full_data["long"].max(), int(np.sqrt(NUM_OF_BINS)))
    # longitude_step = abs(full_data["long"].max - full_data["long"].min) // np.sqrt(NUM_OF_BINS)
    latitude_bins = np.linspace(full_data["lat"].min(), full_data["lat"].max(), int(np.sqrt(NUM_OF_BINS)))
    # latitude_step = abs(full_data["lat"].max - full_data["lat"].min) // np.sqrt(NUM_OF_BINS)
    # groups = full_data.groupby(pd.cut(full_data.long, longitude_bins))
    area_grid = np.array(np.meshgrid(longitude_bins, latitude_bins)).T.reshape(int(np.sqrt(NUM_OF_BINS)),int(np.sqrt(NUM_OF_BINS)), 2)
    print(area_grid)



    pd.DataFrame(data).to_csv("./temp_prices.csv")
    response = full_data["price"]
    return data.to_numpy(), response.to_numpy()


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
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("../datasets/house_prices.csv")
    # print(response)
    # print(data.shape)

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    reg = LinearRegression(False)
    X = np.arange(100).reshape(10, 10)
    y = np.arange(10)
    reg.fit(X, y)
    reg.loss(X, y)

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
