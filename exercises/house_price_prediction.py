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

NUM_OF_AREA_BINS = 36  # should be a power of an int

TOTAL_AVERAGE = 0
AREA_BINS = np.zeros(NUM_OF_AREA_BINS)

"""
date -> year.month
renovated -> time since renovated : today.year - max(yr_built, yr_renovated)

long-lat -> average in section
            if new sample arrives and doesnt have bin, it gets the total average

"""


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
    full_data = full_data[full_data["price"] > 0]

    # remove empty price
    full_data = full_data[full_data["price"].notna()]


    # remove 0 or empty zipcode
    full_data = full_data[full_data["zipcode"].notna()]
    full_data = full_data[full_data["zipcode"] > 0]

    # remove sqft under 0
    for col in full_data.head():
        if ("sqft" in col):
            full_data = full_data[full_data[str(col)] >= 0]



    # remove empty date
    full_data = full_data[full_data["date"].notna()]


    data = full_data.drop(DROP, axis=1)


    # Convert date to year.month 201412 -> 2014.12
    data["date"] = full_data["date"].apply(
        lambda x: pd.Timestamp(x).year + (pd.Timestamp(x).month / 10)
    )
    #  replace missing with mean
    # data["date"].fillna(data["date"].mean(), inplace=True)



    # add time since renovated : today.year - max(yr_built, yr_renovated)
    data["since_renovated"] = pd.DataFrame(
        full_data[["yr_built", "yr_renovated"]].max(axis=1)
        - pd.Timestamp.today().year).abs()

    # create long lat to areas bins

    longitude_bins = np.linspace(full_data["long"].min(), full_data["long"].max(), int(np.sqrt(NUM_OF_AREA_BINS)))
    latitude_bins = np.linspace(full_data["lat"].min(), full_data["lat"].max(), int(np.sqrt(NUM_OF_AREA_BINS)))
    area_grid = np.array(np.meshgrid(longitude_bins, latitude_bins)).T.reshape(int(np.sqrt(NUM_OF_AREA_BINS)),
                                                                               int(np.sqrt(NUM_OF_AREA_BINS)), 2)

    def get_area_bin(sample: pd.DataFrame) -> str:
        long = sample["long"]
        lat = sample["lat"]
        i, j, bin_num = 0, 0, 0
        while i < area_grid.shape[0] and j < area_grid.shape[1]:
            if long > area_grid[i][j][0]:
                i += 1
            if lat > area_grid[i][j][1]:
                j += 1
            if long <= area_grid[i][j][0] and lat <= area_grid[i][j][1]:
                break  # in correct bin
        bin_num = i * area_grid.shape[0] + j
        return f"area_bin_{bin_num}"

    full_data["area_bin"] = full_data[["long", "lat"]].T.apply(get_area_bin)
    area_bins = pd.get_dummies(full_data["area_bin"])

    # add missing area bins
    count = 0
    for i in range(NUM_OF_AREA_BINS):
        if f"area_bin_{i}" not in area_bins.head():
            count += 1
            area_bins[f"area_bin_{i}"] = 0
    print(count)
    area_bins = area_bins[[f"area_bin_{i}" for i in range(NUM_OF_AREA_BINS)]]  # sorted
    data = pd.concat([data, area_bins], axis=1)

    # convert zip to zip_code_bins
    zipcode_bins = pd.get_dummies(full_data["zipcode"])
    data = pd.concat([data, zipcode_bins], axis=1)

    #


    pd.DataFrame(data).to_csv("./temp_prices.csv")

    response = full_data["price"]

    # # replace missing value with mean
    # response.fillna(response.mean(), inplace=True)
    return data, response


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
    train_X, train_y, test_X, test_y = split_train_test(data, response)
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = np.insert(test_X.to_numpy(), 0, 1, axis=1)
    test_y = test_y.to_numpy()
    # print(pd.DataFrame(train_X).isnull().values.any())
    # print(pd.DataFrame(train_y).isnull().values.any())
    # print(pd.DataFrame(test_X).isnull().values.any())
    # print(pd.DataFrame(test_y).isnull().values.any())
    # test_X = np.insert(test_X.to_numpy(), 0, 1, axis=1)
    # pd.DataFrame(test_X).insert(0, "bias", 1)
    # test_X = test_X["bias"]



    # Fit model over data
    reg = LinearRegression()
    reg.fit(train_X, train_y)
    res = reg.predict(test_X)
    res = pd.DataFrame(res, columns=["pred_y"])
    res["test_y"] = test_y
    res["error_prec"] = (res["pred_y"] - res["test_y"]).abs() / res["test_y"].abs()
    print(res["error_prec"].mean())
    pd.DataFrame(res).to_csv("./temp_res.csv")

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    X = np.arange(100).reshape(10, 10)
    y = np.arange(10)
    # reg.fit(X, y)
    # reg.loss(X, y)

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
