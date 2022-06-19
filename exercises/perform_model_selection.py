from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.io
from sklearn import datasets

from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plotly.io.renderers.default = 'browser'


class LassoDummy(BaseEstimator):
    def __init__(self, lam: float):
        super().__init__()
        self._lam = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.lasso = Lasso(self._lam, max_iter=2000)
        self.lasso.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.lasso.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return mean_square_error(self.lasso.predict(X), y)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def f(x: int):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = list(map(f, X))
    y = y_noiseless + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y), train_proportion=2 / 3)

    train_X, train_y, test_X, test_y = train_X.sort_index().iloc[:, 0].to_numpy(), \
                                       train_y.sort_index().to_numpy(), \
                                       test_X.sort_index().iloc[:, 0].to_numpy(), \
                                       test_y.sort_index().to_numpy()

    # plot noiseless vs noised data
    go.Figure(
        data=[go.Scatter(x=X, y=y_noiseless, name='true noiseless', mode='markers'),
              go.Scatter(x=train_X, y=train_y, name='train with noise', mode='markers'),
              go.Scatter(x=test_X, y=test_y, name='test with noise', mode='markers')],
        layout=go.Layout(
            title=fr'$\text{{True vs Test and Train sets with normal noise }} \sigma^2 = {noise}$',
            xaxis_title='x',
            yaxis_title='y',
            height=1000,
            width=1000)
    ).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = np.zeros(11)
    validation_scores = np.zeros(11)
    for k in range(11):
        train_scores[k], validation_scores[k] = cross_validate(PolynomialFitting(k),
                                                               train_X,
                                                               train_y,
                                                               mean_square_error)
    go.Figure(
        data=[
            go.Scatter(
                x=list(range(11)),
                y=validation_scores,
                name='Validation',
                mode='markers'
            ),
            go.Scatter(
                x=list(range(11)),
                y=train_scores,
                name='Train',
                mode='markers'
            )
        ],
        layout=go.Layout(
            title=fr'$\text{{Validation and Train Scores for data with normal noise }} \sigma^2 = {noise}$',
            xaxis_title='Polynomial Degree',
            yaxis_title='Loss',
            xaxis=dict(dtick=1),
            height=1000,
            width=1000)
    ).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = int(np.argmin(validation_scores))
    polyfit = PolynomialFitting(k)
    pred_y = polyfit.fit(train_X, train_y).predict(test_X)
    test_error = mean_square_error(test_y, pred_y)
    # print()
    print(f"k^{k}, test error = {round(test_error, 2)} for {n_samples} samples with noise {noise}")

    # go.Figure(
    #     data=[go.Scatter(x=X, y=y_noiseless, name='true noiseless', mode='markers'),
    #           go.Scatter(x=train_X, y=polyfit.predict(train_X), name='train prediction', mode='markers'),
    #           go.Scatter(x=test_X, y=pred_y, name='test prediction', mode='markers')],
    #     layout=go.Layout(
    #         title=fr'$\text{{True vs Test and Train sets predictions for data with normal noise }} \sigma^2 = {noise}$',
    #         xaxis_title='x',
    #         yaxis_title='y',
    #         height=1000,
    #         width=1000)
    # ).show()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], \
                                       X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_min = 0.001
    lambda_max = 3
    lambda_range = np.linspace(lambda_min, lambda_max, n_evaluations)

    train_er_ind = 0
    val_er_ind = 1

    ridge_scores = np.zeros(shape=(n_evaluations, 2))
    lasso_scores = np.zeros(shape=(n_evaluations, 2))

    for i, lam in enumerate(lambda_range):
        ridge_scores[i] = cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(LassoDummy(lam), train_X, train_y, mean_square_error)

    scores = [ridge_scores, lasso_scores]
    titles = ["Ridge", "Lasso"]
    fig = make_subplots(rows=1, cols=2,
                        column_titles=[
                            fr'$\text{{Validation and Train Scores for {titles[i]} model }}$'
                            for i in range(2)])
    for i in range(2):
        fig.add_traces(
            data=[
                go.Scatter(
                    x=lambda_range,
                    y=scores[i][:, val_er_ind],
                    name=f'{titles[i]} Validation',
                    mode='markers'
                ),
                go.Scatter(
                    x=lambda_range,
                    y=scores[i][:, train_er_ind],
                    name=f'{titles[i]} Train',
                    mode='markers'
                )
            ],
            rows=1,
            cols=i % 2 + 1
        )

    fig.update_layout(xaxis_title=r"$\lambda$",
                      yaxis_title=r"$\text{Loss}$", )

    fig.show()

    best_lasso = lambda_range[np.argmin(lasso_scores[:, val_er_ind])]

    best_ridge = lambda_range[np.argmin(ridge_scores[:, val_er_ind])]

    print(
        f"Lasso best lambda = {best_lasso}\n" +
        f"Ridge best lambda = {best_ridge}"
    )

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    lasso = LassoDummy(lam=best_lasso).fit(train_X, train_y)
    ridge = RidgeRegression(lam=best_ridge).fit(train_X, train_y)
    lin = LinearRegression().fit(train_X, train_y)
    print(
        f"Lasso Loss over Test: {lasso.loss(test_X, test_y)}\n" +
        f"Ridge Loss over Test {ridge.loss(test_X, test_y)}\n"+
        f"Least Squares Loss over Test {lin.loss(test_X, test_y)}"
    )


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
