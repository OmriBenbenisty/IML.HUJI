import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import plotly.graph_objects as go
pio.templates.default = "simple_white"

pio.renderers.default = "browser"


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
    date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y')
    full_data = pd.read_csv(filename, parse_dates=["Date"])

    # remove extreme temp's
    full_data = full_data[full_data["Temp"] > -15]

    # add day of year
    full_data["day_of_year"] = full_data["Date"].dt.dayofyear

    # full_data.to_csv("./temp_city_temp.csv")

    return full_data



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    data = data[data["Country"] == 'Israel']

    # go.Figure(
    #     [go.Scatter(
    #         x=data["day_of_year"],
    #         y=data["Temp"],
    #         mode='markers',
    #         name='Temperature in Israel Per Day Of Year'
    #     )],
    #     layout= go.Layout(
    #         title={"text": f"Temperature in Israel As Function Of Day Of Year",
    #                "x": 0.5,
    #                "xanchor": 'center'},
    #         titlefont={'family': 'Arial',
    #                    'size': 30},
    #         xaxis_title={'text': "Day of Year",
    #                      'font': {'family': 'Arial',
    #                               'size': 20}},
    #         yaxis_title={'text': "Temperature",
    #                      'font': {'family': 'Arial',
    #                               'size': 20}},
    #         yaxis=dict(dtick=1),
    #         width=1920,
    #         height=1080
    #     )
    # ).show()

    month_std_groups = data.groupby("Month").std()["Temp"]
    px.bar(month_std_groups,"Month",)
    go.Figure(
        [go.Bar],
        layout=go.Layout(

        )
    )
    print(month_std_groups)



    # Question 3 - Exploring differences between countries

    # Question 4 - Fitting model for different values of `k`

    # Question 5 - Evaluating fitted model on different countries
