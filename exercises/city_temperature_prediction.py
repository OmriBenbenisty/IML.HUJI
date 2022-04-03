from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
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
    full_data = pd.read_csv(filename, parse_dates=["Date"])

    # remove extreme temp's
    full_data = full_data[full_data["Temp"] > -15]

    # add day of year
    full_data["day_of_year"] = full_data["Date"].dt.dayofyear

    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data["Country"] == 'Israel']

    # plot graphs

    # Temperature in Israel As Function Of Day Of Year
    go.Figure(
        [go.Scatter(
            x=israel_data["day_of_year"],
            y=israel_data["Temp"],
            mode='markers',
            name='Temperature in Israel Per Day Of Year'
        )],
        layout=go.Layout(
            title={"text": f"Temperature in Israel As Function Of Day Of Year",
                   "x": 0.5,
                   "xanchor": 'center'},
            titlefont={'family': 'Arial',
                       'size': 30},
            xaxis_title={'text': "Day of Year",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            yaxis_title={'text': "Temperature",
                         'font': {'family': 'Arial',
                                  'size': 20}},
            yaxis=dict(dtick=1),
            width=1500,
            height=800
        )
    ).show()

    # Temperature STD As a Function of Month in Israel
    month_std_groups = israel_data.groupby("Month").std()
    px.bar(
        data_frame=month_std_groups,
        x=month_std_groups.index,
        y="Temp",
        labels={'Temp': 'Temperature STD'}
    ).update_layout(
        xaxis=dict(dtick=1),
        title={"text": f"Temperature STD As a Function of Month in Israel",
               "x": 0.5,
               "xanchor": 'center'},
        titlefont={'family': 'Arial',
                   'size': 30},
        xaxis_title={'text': "Month",
                     'font': {'family': 'Arial',
                              'size': 20}},
        yaxis_title={'text': "Temperature STD",
                     'font': {'family': 'Arial',
                              'size': 20}},
        yaxis=dict(dtick=1),
        width=1000,
        height=500
    ).show()

    # Question 3 - Exploring differences between countries

    country_month_mean = data.groupby(["Country", "Month"], as_index=False). \
        agg(avg_temp=('Temp', np.mean))

    country_month_std = data.groupby(["Country", "Month"]). \
        agg(temp_std=('Temp', np.std))

    # Average Temperature As a Function of Months
    px.line(
        data_frame=country_month_mean,
        x=country_month_mean["Month"],
        y=country_month_mean["avg_temp"],
        labels={"avg_temp": "Average Temperature"},
        color=country_month_mean["Country"],
        error_y=country_month_std["temp_std"]
    ).update_layout(
        xaxis=dict(dtick=1),
        title={"text": f"Average Temperature As a Function of Months",
               "x": 0.5,
               "xanchor": 'center'},
        titlefont={'family': 'Arial',
                   'size': 30},
        xaxis_title={'text': "Months",
                     'font': {'family': 'Arial',
                              'size': 20}},
        yaxis_title={'text': "Average Temperature",
                     'font': {'family': 'Arial',
                              'size': 20}},
        width=1000,
        height=500
    ).show()

    # Question 4 - Fitting model for different values of `k`

    train_X, train_y, test_X, test_y = split_train_test(
        israel_data["day_of_year"],
        israel_data["Temp"])
    train_X, train_y, test_X, test_y = (train_X.to_numpy(),
                                        train_y.to_numpy(),
                                        test_X.to_numpy(),
                                        test_y.to_numpy())
    loss = np.zeros(10)
    for k in range(1, 11):
        k_poly_fit = PolynomialFitting(k).fit(train_X, train_y)
        loss[k - 1] = round(k_poly_fit.loss(test_X, test_y), 2)
    for i, l in enumerate(loss):
        print(f"PolyFit of Degree {i + 1} Loss = {l}")

    # Loss Over Polynomial Degree
    px.bar(loss,
           x=np.arange(10) + 1,
           y=loss,

           ).update_layout(
        xaxis=dict(dtick=1),
        title={"text": f"Loss Over Polynomial Degree",
               "x": 0.5,
               "xanchor": 'center'},
        titlefont={'family': 'Arial',
                   'size': 30},
        xaxis_title={'text': "Degree",
                     'font': {'family': 'Arial',
                              'size': 20}},
        yaxis_title={'text': "Mean Loss",
                     'font': {'family': 'Arial',
                              'size': 20}},
        width=1000,
        height=500
    ).show()

    # Question 5 - Evaluating fitted model on different countries

    poly_fit_5 = PolynomialFitting(5).fit(israel_data["day_of_year"], israel_data["Temp"])
    country_loss = pd.DataFrame({'Country': ["The Netherlands",
                                             "South Africa",
                                             "Jordan"],
                                 'Loss': [0, 0, 0]})
    temp_loss = np.zeros(3)
    for i, country in enumerate(country_loss['Country']):
        country_data = data[data['Country'] == country]
        temp_loss[i] = poly_fit_5.loss(country_data["day_of_year"].to_numpy(),
                                       country_data['Temp'].to_numpy())
    country_loss['Loss'] = temp_loss

    px.bar(data_frame=country_loss,
           x=country_loss['Country'],
           y=country_loss['Loss']
           ).update_layout(
        title={"text": f"Loss Over Different Countries<br> <sup> With A Polynomial Fit For Israel With Degree of 5</sup>",
               "x": 0.5,
               "xanchor": 'center'},
        titlefont={'family': 'Arial',
                   'size': 30},
        xaxis_title={'text': "Country",
                     'font': {'family': 'Arial',
                              'size': 20}},
        yaxis_title={'text': "Mean Loss",
                     'font': {'family': 'Arial',
                              'size': 20}},
        width=1300,
        height=700
    ).show()
