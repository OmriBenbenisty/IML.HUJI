import numpy as np
from typing import Tuple

import plotly.io

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plotly.io.renderers.default = 'browser'


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    print("Fitting.......")
    adb = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    print("Plotting.......")
    go.Figure(
        data=[
            go.Scatter(
                x=list(range(1, n_learners + 1)),
                y=list(map(lambda n: adb.partial_loss(train_X, train_y, n), list(range(1, n_learners + 1)))),
                mode='markers+lines',
                name="Training Loss"
            ),
            go.Scatter(
                x=list(range(1, n_learners + 1)),
                y=list(map(lambda n: adb.partial_loss(test_X, test_y, n), list(range(1, n_learners + 1)))),
                mode='markers+lines',
                name="Test Loss"
            )
        ],
        layout=go.Layout(
            title="Loss as Function of Num of Learners",
            xaxis_title={'text': "$\\text{Num of Learners}$"},
            yaxis_title={'text': "$\\text{Misclassification Loss}$"}
        )
    ).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    preds = [adb.partial_predict(train_X, t) for t in T]
    symbols = np.array(["circle", "x", "diamond"])
    title = ""

    fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=[f"Decision Boundary for Ensemble of Size {m}"
                                        for i, m in enumerate(T)],
                        horizontal_spacing=0.1,
                        vertical_spacing=.03,
                        )

    # Add traces for data-points setting symbols and colors
    for i, m in enumerate(T):
        fig.add_traces([go.Scatter(
            x=train_X[:, 0],
            y=train_X[:, 1],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=preds[i],
                symbol=symbols[train_y.astype(int)],
                colorscale=[custom[0], custom[-1]],
                line=dict(color="black", width=1)),
        ),
            decision_surface(lambda x: adb.partial_predict(x, m), lims[0], lims[1], showscale=False)

        ],
            rows=(i // 2) + 1, cols=(i % 2) + 1
        )

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries for Different Ensemble Size$",
        # margin=dict(t=100),
        width=1200,
        height=1000
    )

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, train_size=5000)
