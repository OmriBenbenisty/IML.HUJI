from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

pio.templates.default = "simple_white"

SAMPLES = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, SAMPLES)
    uge = UnivariateGaussian()
    uge.fit(X)
    print(uge.mu_, uge.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, SAMPLES, 100).astype(int)
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
              layout=go.Layout(title=f"{SAMPLES} Random Normal Samples PDF Values Scatter",
                               xaxis_title=r"$\text{Sample Value}$",
                               yaxis_title=r"PDF")).show()


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
    f1 = np.linspace(-10, 10, space_size)
    f3 = np.linspace(-10, 10, space_size)
    mus = np.array(np.meshgrid(f1, f3)).T.reshape(space_size, space_size, 2)

    # z = np.array([[mge.log_likelihood(mu, cov, X) for mu in mus[i]] for i in range(mus.shape[0])]).T

    def calc_mu(m):
        return mge.log_likelihood(np.array([m[0], 0, m[1], 0]), cov, X)

    z = np.array([list(map(calc_mu, mus[i])) for i in range(mus.shape[0])])
    fig = go.Figure(data=go.Heatmap(x=f3, y=f1, z=z),
                    layout=go.Layout(
                        title='Log-Likelihood of Multivariate Normal Distribution with True Expectation of 0, 0, 4, 0' +
                              '<br>' +
                              'as a Function of f1, 0, f3, 0 Over 1000 Samples',
                        xaxis_title=r"$\text{f3}$",
                        yaxis_title=r"$\text{f1}$",

                        xaxis=dict(dtick=1),
                        yaxis=dict(dtick=1),
                        height=800,
                        width=800)
                    )
    fig.show()

    # Question 6 - Maximum likelihood
    coord = np.where(z == np.amax(z))
    print(mus[coord])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
