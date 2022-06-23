import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.metrics
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error, mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from utils import custom
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly as plt

plt.io.renderers.default = 'browser'

titles = ["L1", "L2"]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs) -> None:
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for j, m in enumerate([L1, L2]):
        fig = go.Figure()
        for i, eta in enumerate(etas):
            callback, values, weights = get_gd_state_recorder_callback()
            module = m(weights=init.copy())
            weights.append(init.copy())
            gd = GradientDescent(callback=callback, learning_rate=FixedLR(eta)).fit(module, None, None)
            plot_descent_path(module=m,
                              descent_path=np.array(weights),
                              title=f"Descent path for {titles[j]} norm with eta = {eta}").show()
            fig.add_traces(data=[go.Scatter(x=np.arange(len(values)), y=values, name=eta)])
            # print(f"{titles[j]}, {eta},min = {round(min(values), 4)}")
        fig.update_layout(
            title=f"{titles[j]} norm as a function of the GD iteration"
        ).show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure()
    for i, gamma in enumerate(gammas):
        callback, values, weights = get_gd_state_recorder_callback()
        module = L1(weights=init.copy())
        weights.append(init.copy())
        gd = GradientDescent(callback=callback, learning_rate=ExponentialLR(eta, gamma)).fit(module, None, None)
        plot_descent_path(module=L1,
                          descent_path=np.array(weights),
                          title=f"Descent path for L1 norm with eta = {eta}, gamma = {gamma}").show()
        fig.add_traces(data=[go.Scatter(x=np.arange(len(values)), y=values, name=gamma, mode='markers')])

        print(f"L1, eta={eta},gamma={gamma} ,min = {round(min(values), 4)}")

    # Plot algorithm's convergence for the different values of gamma
    fig.update_layout(
        title=f"L1 norm as a function of the GD iteration with Exponential Decay eta = {eta}",
        xaxis_title="Iterations",
        yaxis_title="L1 Norm"
    ).show()

    # Plot descent path for gamma=0.95


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression(
        solver=GradientDescent(
            learning_rate=FixedLR(1e-4),
            max_iter=20000)
    ).fit(
        X=X_train,
        y=y_train)
    pred = lr.predict_proba(X_train)

    c = [custom[0], custom[-1]]
    fpr, tpr, thresholds = roc_curve(y_train, pred)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    alpha_i = np.argmax(tpr - fpr)
    alpha_best = thresholds[alpha_i]

    test_error = misclassification_error(
        y_true=y_test,
        y_pred=lr.predict_proba(X_test) >= alpha_best
    )

    print(f"Best Alpha = {alpha_best}, with test error = {test_error}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # lambda_min = 0.001
    # lambda_max = 0.2
    # n_evaluations = 10
    # lambda_range = np.linspace(lambda_min, lambda_max, n_evaluations)

    train_er_ind = 0
    val_er_ind = 1
    alpha = 0.5
    lambda_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    n_evaluations = len(lambda_range)

    for penalty in ["l1", "l2"]:
        scores = np.zeros(shape=(n_evaluations, 2))
        for i, lam in enumerate(lambda_range):
            reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                            max_iter=20000),
                                     penalty=penalty,
                                     lam=lam,
                                     alpha=alpha
                                     )
            scores[i] = cross_validate(estimator=reg, X=X_train, y=y_train, scoring=mean_square_error)

        go.Figure(
            data=[
                go.Scatter(
                    x=lambda_range,
                    y=scores[:, val_er_ind],
                    name=f'{penalty} penalty Validation',
                    mode='markers+lines'
                ),
                go.Scatter(
                    x=lambda_range,
                    y=scores[:, train_er_ind],
                    name=f'{penalty} penalty Train',
                    mode='markers+lines'
                )
            ]
        ).show()
        best_lam = lambda_range[np.argmin(scores[:, val_er_ind])]
        print(f"Best Lam for {penalty} penalty = {best_lam}")

        module = LogisticRegression(
            penalty=penalty,
            solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
            lam=best_lam
        )
        module.fit(X_train, y_train)
        test_error = misclassification_error(y_true=y_test, y_pred=module.predict(X_test))
        print(f"Penalty {penalty} With Lam = {best_lam} Test error = {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
    print("Done")

