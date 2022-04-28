import numpy as np
import plotly.io

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2, pi

plotly.io.renderers.default = "browser"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda p, x, y_i: losses.append(p.loss(X, y))).fit(X, y)

        # plot the data
        # px.scatter(
        #     x=X[:, :1].reshape(X.shape[0]),
        #     y=X[:, 1].reshape(X.shape[0]),
        #     color=y.astype(str)
        # ).show()

        # Plot figure of loss as function of fitting iteration
        go.Figure(
            data=[go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                mode='markers+lines')
            ],
            layout=go.Layout(title="Loss as Function of Fitting Iteration<br>" +
                                   f"Over {n} Data",
                             xaxis_title={'text': "$\\text{Fitting Iteration}$"},
                             yaxis_title={'text': "$\\text{Misclassification Loss}$"})
        ).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray) -> go.Scatter:
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for j, f in enumerate(["gaussian1.npy", "gaussian2.npy"]):
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        preds = [gnb.predict(X), lda.predict(X)]
        print(np.round(lda.likelihood(X), 4))
        print("\n\n\n")
        continue

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        models = [gnb, lda]
        model_names = ["GaussianNaiveBayes", "LDA"]
        symbols = np.array(["circle", "x", "diamond"])
        title = "Gaussian"

        # Create subplots
        fig = make_subplots(rows=2,
                            cols=2,
                            subplot_titles=[f"{m} <br><sup>Accuracy = {round(accuracy(y, preds[i]), 3)}</sup>"
                                            for i, m in enumerate(model_names)],
                            horizontal_spacing=0.1,
                            vertical_spacing=.03,
                            )

        # Add traces for data-points setting symbols and colors
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=preds[i],
                                                   symbol=symbols[y],
                                                   colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)),
                                       ),
                            decision_surface(m.predict, lims[0], lims[1], showscale=False)

                            ],
                           rows=(i // 2) + 1, cols=(i % 2) + 1
                           )

        fig.update_layout(
            title=rf"$\textbf{{Decision Boundaries Of Models - {title} {j + 1} Dataset}}$",
            margin=dict(t=100),
            width=1200,
            height=1000
        )

        # fig.show()

        # Add `X` dots specifying fitted Gaussians' means
        colors = ['red', 'yellow', 'blue']
        for i, m in enumerate(models):
            for s, mu in enumerate(m.mu_):
                fig.add_traces([
                    go.Scatter(
                        x=[mu[0]],
                        y=[mu[1]],
                        mode='markers',
                        showlegend=False,
                        marker=dict(
                            color=colors[s],
                            symbol='x-dot',
                            colorscale=[custom[0], custom[-1]]
                        )
                    )
                ],
                    rows=(i // 2) + 1, cols=(i % 2) + 1
                )

        # Add ellipses depicting the covariances of the fitted Gaussians
        covs = np.empty(shape=(np.unique(y).shape[0], X.shape[1], X.shape[1]))
        for l, k in enumerate(np.unique(y)):
            covs[l] = np.cov(X[y == k].T)
        # print(f"{title} {j + 1} covs = \n {np.round(covs, 3)}")

        for i, m in enumerate(models):
            for mu_ind, mu in enumerate(m.mu_):
                # mu = m.mu_[j]
                # cov = np.cov(X[y==k].T)
                fig.add_traces([
                    get_ellipse(mu, covs[mu_ind] if m == lda else np.diag(m.vars_[mu_ind]))
                ],
                    rows=(i // 2) + 1, cols=(i % 2) + 1
                )

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()

    # X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    # y = np.array([ 0,  0,  1,  1,  1,  1])
    # gnb = GaussianNaiveBayes().fit(X, y)
    # print(gnb.vars_[0,0])
    #
    # print(gnb.vars_)
    # print("done")
