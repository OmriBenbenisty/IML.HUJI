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

"""
date -> year.month
renovated -> time since renovated : today.year - max(yr_built, yr_renovated)

long-lat -> area bins
            if new sample arrives and doesnt have bin, its assigned to the
            closest bin
zip -> zip code dummies
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
    for feature in full_data.head():
        if "sqft" in feature:
            full_data = full_data[full_data[str(feature)] >= 0]

    # remove empty date  *** Skipped duo to low correlation ***
    # full_data = full_data[full_data["date"].notna()]

    data = full_data.drop(DROP, axis=1)

    # Convert date to year.month 201412 -> 2014.12 *** Skipped duo to low correlation ***
    # data["date"] = full_data["date"].apply(
    #     lambda x: pd.Timestamp(x).year + (pd.Timestamp(x).month / 10)
    # )

    # add time since renovated : today.year - max(yr_built, yr_renovated)
    data["since_renovated"] = pd.DataFrame(
        full_data[["yr_built", "yr_renovated"]].max(axis=1)
        - pd.Timestamp.today().year).abs()

    # create long lat to areas bins
    longitude_bins = np.linspace(full_data["long"].min(), full_data["long"].max(),
                                 int(np.sqrt(NUM_OF_AREA_BINS)))
    latitude_bins = np.linspace(full_data["lat"].min(), full_data["lat"].max(),
                                int(np.sqrt(NUM_OF_AREA_BINS)))
    area_grid = np.array(np.meshgrid(longitude_bins, latitude_bins)).T.reshape(
        int(np.sqrt(NUM_OF_AREA_BINS)), int(np.sqrt(NUM_OF_AREA_BINS)), 2)

    def get_area_bin(sample: pd.DataFrame) -> str:
        long = sample["long"]
        lat = sample["lat"]
        row, col, bin_num = 0, 0, 0
        while row < area_grid.shape[0] and col < area_grid.shape[1]:
            if long > area_grid[row][col][0]:
                row += 1
            if lat > area_grid[row][col][1]:
                col += 1
            if long <= area_grid[row][col][0] and lat <= area_grid[row][col][1]:
                break  # correct bin
        bin_num = row * area_grid.shape[0] + col
        return f"area_bin_{bin_num}"

    full_data["area_bin"] = full_data[["long", "lat"]].T.apply(get_area_bin)
    area_bins = pd.get_dummies(full_data["area_bin"])

    # add missing area bins
    for j in range(NUM_OF_AREA_BINS):
        if f"area_bin_{j}" not in area_bins.head():
            area_bins[f"area_bin_{j}"] = 0
    area_bins = area_bins[[f"area_bin_{bin_num}" for bin_num in range(NUM_OF_AREA_BINS)]]  # sorted
    data = pd.concat([data, area_bins], axis=1)

    # convert zip to zip_code_bins
    zipcode_bins = pd.get_dummies(full_data["zipcode"], prefix='zipcode_')
    data = pd.concat([data, zipcode_bins], axis=1)

    response = full_data["price"]

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
    X_arr = X.to_numpy().T
    y = y.to_numpy()

    def corr_coef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if np.var(x) == 0 or np.var(y) == 0:
            return np.zeros(4).reshape((2, 2))
        return np.cov(x, y) / np.sqrt(np.var(x) * np.var(y))

    corr = np.array([corr_coef(X_arr[i], y)[0, 1] for i in range(X_arr.shape[0])]).reshape((X_arr.shape[0]))
    for i, c in enumerate(corr):
        if "area" not in str(X.columns[i]) and "zipcode" not in str(X.columns[i]):
            go.Figure([go.Scatter(x=X_arr[i], y=y, mode='markers', name=r'$Response$')],
                      layout=go.Layout(
                          title={"text": f"Feature - {X.columns[i]} <br><sup>Pearson Correlation - {c}</sup>",
                                 "x": 0.5,
                                 "xanchor": 'center'},
                          titlefont={'family': 'Arial',
                                     'size': 30},
                          xaxis_title={'text': "Feature Value",
                                       'font': {'family': 'Arial',
                                                'size': 20}},
                          yaxis_title={'text': "Response",
                                       'font': {'family': 'Arial',
                                                'size': 20}},
                          width=1000,
                          height=1000
                      )
                      ).write_image(f"{output_path}/{X.columns[i]}_graph.png")
    # zip_mean = [corr[i] for i in range(X_arr.shape[0]) if "zipcode" in X.columns[i]]
    # print(np.mean(zip_mean))
    # area_mean = [corr[i] for i in range(X_arr.shape[0]) if "area" in X.columns[i]]
    # print(np.mean(area_mean))


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(data, response, "./Images")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, response)
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    reg = LinearRegression()
    repeat = 10
    mean_std = np.zeros(182).reshape((91, 2))
    for p in range(10, 101):
        loss = np.zeros(repeat)
        for i in range(repeat):
            X = train_X.sample(frac=p / 100)
            y = train_y[X.index]
            reg.fit(X.to_numpy(), y.to_numpy())
            loss[i] = reg.loss(test_X, test_y)
        mean_std[p - 10][0], mean_std[p - 10][1] = loss.mean(), loss.std()
    y_top = mean_std[:, 0] + (2 * mean_std[:, 1])
    y_min = mean_std[:, 0] - (2 * mean_std[:, 1])
    go.Figure([
        go.Scatter(
            x=list(range(10, 101)),
            y=mean_std[:, 0],
            mode='markers+lines',
            line=dict(color='rgb(0,100,80)'),
            name="Loss Mean Over 10 Repetitions"
        ),
        go.Scatter(
            x=list(range(10, 101)) + list(range(10, 101))[::-1],
            y=np.concatenate((y_top, y_min[::-1])),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name=r"$\text{Loss Mean} \pm \text{2 * Loss STD Over 10 Repetitions}$"
        )],
        layout=go.Layout(
            title={"text": f"Loss Over Test Set As a Function of Train Set Percentage",
                   "x": 0.5,
                   "xanchor": 'center'},
            titlefont={'family': 'Arial',
                       'size': 30},
            xaxis_title={'text': "Train Set Percentage",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            yaxis_title={'text': "Loss Over Test Set",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            width=2000,
            height=1000
        )
    ).show()
