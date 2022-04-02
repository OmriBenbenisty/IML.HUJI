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

# TOTAL_AVERAGE = 0
# AREA_BINS = np.zeros(NUM_OF_AREA_BINS)

"""
date -> year.month
renovated -> time since renovated : today.year - max(yr_built, yr_renovated)

long-lat -> average in section
            if new sample arrives and doesnt have bin, it gets the total average
zip
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
    for i in range(NUM_OF_AREA_BINS):
        if f"area_bin_{i}" not in area_bins.head():
            area_bins[f"area_bin_{i}"] = 0
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
    X_arr = X.to_numpy().T
    y = y.to_numpy()

    def corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if np.var(x) == 0 or np.var(y) == 0:
            return np.zeros(4).reshape((2, 2))
        return np.cov(x, y) / np.sqrt(np.var(x) * np.var(y))

    # corr = lambda x, y: np.cov(x, y) / np.sqrt(np.var(x) * np.var(y))
    # corr = np.array([np.corrcoef(X_arr[i], y)[0,1] for i in range(X_arr.shape[0])]).reshape((X_arr.shape[0]))
    corr = np.array([corr(X_arr[i], y)[0, 1] for i in range(X_arr.shape[0])]).reshape((X_arr.shape[0]))
    for i, c in enumerate(corr):
        if "area" not in str(X.columns[i]) and "98" not in str(X.columns[i]):
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


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, response = load_data("../datasets/house_prices.csv")

    # print(pd.DataFrame(train_X).isnull().values.any())
    # print(pd.DataFrame(train_y).isnull().values.any())
    # print(pd.DataFrame(test_X).isnull().values.any())
    # print(pd.DataFrame(test_y).isnull().values.any())
    # test_X = np.insert(test_X.to_numpy(), 0, 1, axis=1)
    # pd.DataFrame(test_X).insert(0, "bias", 1)
    # test_X = test_X["bias"]

    # Fit model over data
    reg = LinearRegression().fit(data.to_numpy(), response.to_numpy())


    # res = pd.DataFrame(res, columns=["pred_y"])
    # res["test_y"] = test_y
    # res["error_prec"] = (res["pred_y"] - res["test_y"]).abs() / res["test_y"].abs()
    # print(res["error_prec"].mean())
    # pd.DataFrame(res).to_csv("./temp_res.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(data, response, "./Images")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, response)
    # train_X = train_X.to_numpy()
    # train_y = train_y.to_numpy()
    # test_X = np.insert(test_X.to_numpy(), 0, 1, axis=1)
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()

    # res = reg.predict(test_X)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    train = pd.concat([train_X, train_y], axis=1)
    repeat = 10
    mean_std = np.zeros(182).reshape((91, 2))
    for p in range(10, 101):
        loss = np.zeros(repeat)
        for i in range(repeat):
            train_p = train.sample(frac=p / 100)
            X, y = train_p.iloc[:, :-1].to_numpy(), train_p.iloc[:, -1:].to_numpy()
            reg.fit(X, y)

            loss[i] = reg.loss(test_X, test_y)
        mean_std[p - 10][0], mean_std[p - 10][1] = loss.mean(), loss.std()
    y_top = mean_std[:, 0] + (2 * mean_std[:, 1])
    y_min = mean_std[:, 0] - (2 * mean_std[:, 1])
    go.Figure([
        go.Scatter(
            x=list(range(10, 101)),
            y=mean_std[:, 0],
            mode='lines',
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
            xaxis_title={'text': "Test Set Percentage",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            yaxis_title={'text': "Loss Over Train Set",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            width=1000,
            height=1300
        )
    ).write_image(f"Images/mean_loss_graph.png")
    print("ended")
