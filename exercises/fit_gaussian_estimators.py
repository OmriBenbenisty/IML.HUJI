import numpy

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, 1000)
    uge = UnivariateGaussian()
    uge.fit(X)
    print(uge.mu_, uge.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    num_of_samples = 100
    estimations = np.zeros((num_of_samples, 2))  # mu, sigma^2
    for i in range(1, num_of_samples):
        uge.fit(X[0:i * 10])
        estimations[i - 1] = uge.mu_, uge.var_

    go.Figure([go.Scatter(x=ms,
                          y=np.abs(estimations[:-1, 0] - ([mu] * (num_of_samples - 1))),
                          mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between the"
                                     r" Estimated and True Value of Expectation"
                                     r" As Function of Sample Size}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\hat\mu-\mu|$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    uge.fit(X)
    go.Figure([go.Scatter(x=X, y=uge.pdf(X), mode='markers', name=r'$pdf$')],
              layout=go.Layout(title=r"$\text{Random Normal Samples PDF Values Scatter}$",
                               xaxis_title=r"$\text{Random Normal(10,1) Samples}$",
                               yaxis_title=r"$PDF$")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)

    mge = MultivariateGaussian()
    mge.fit(X)
    print(f"{mge.mu_}\n{mge.cov_}")

    # Question 5 - Likelihood evaluation
    space_size = 200
    f1 = np.linspace(-13, 10, space_size)
    f3 = np.linspace(-10, 10, space_size)
    # mus = np.stack((f1, np.zeros(space_size), f3, np.zeros(space_size)), 1)
    # mus = np.fromiter((np.ndarray([i, 0, j, 0]) for i in f1 for j in f3), dtype=np.float64)
    # mus = np.zeros((space_size, space_size, 4))
    # z = np.empty((space_size, space_size))
    # for i, val1 in enumerate(f1):
    #     for j, val2 in enumerate(f3):
            # print(val1)
            # mus[i, j] = np.array([val1, 0, val2, 0])
            # m = np.array([val1, 0, val2, 0])
            # z[i, j] = mge.log_likelihood(m, cov, X)
            # print(mge.log_likelihood(m, cov, X))
    mus = np.array(np.meshgrid(f1, [0], f3, [0])).T.reshape(space_size, space_size, 4)
    z = np.array([[mge.log_likelihood(m, cov, X) for m in mus[i]] for i in range(mus.shape[0])]).T  # good
    # z = np.fromfunction(lambda i, j: mge.log_likelihood(mus[i, j], cov, X), (space_size, space_size), dtype=int)
    def calc(mu):
        return mge.log_likelihood(mu, cov, X)
    # z = [list(map(calc, mus[i])) for i in range(space_size)]
    # z = np.fromfunction(lambda i, j: j, (space_size, space_size), dtype=int)
    # z = np.fromfunction(lg_likelihood, (space_size, space_size))
    # z = np.fromfunction(lambda i, j: print(type(i[0][0])), (space_size, 1))
    # print(z)
    # fig = go.Figure(data=go.Heatmap(x=f3, y=f1, z=z))
    # fig = go.Figure(data=go.Heatmap(x=f3, y=f1, z=z), layout=go.Layout(
    #     title=r"$\text{Log-Likelihood of Multivariate Normal Distribution with Expectation of 0, 0, 4, 0}"
    #           r"\r\text{            f1, 0, f3, 0}$",
    #     xaxis_title=r"$\text{Expectation Values}$",
    #     yaxis_title=r"$\text{Expectation Values}$"))
    fig = go.Figure(data=go.Heatmap(x=f3, y=f1, z=z), layout=go.Layout(
        title='Log-Likelihood of Multivariate Normal Distribution with true Expectation of 0, 0, 4, 0' +
              '<br>' +
              'and Log-Likelihood as a Function of f1, 0, f3, 0',
        xaxis_title=r"$\text{f3}$",
        yaxis_title=r"$\text{f1}$"))
    fig.show()
    # ,
    #         zaxis_title=r"$\text{Log-Likelihood}$"

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
