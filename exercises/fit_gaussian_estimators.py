from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from utils import *

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    requested_mu = 10
    requested_sigma = 1
    X = np.linspace(1, 1000, num=1000)
    Y = np.random.normal(requested_mu, requested_sigma, size=1000)
    Z = UnivariateGaussian().fit(Y)
    mu_hat, sigma_hat = Z.mu_, Z.var_
    print(mu_hat, sigma_hat)

    # make_subplots(rows=1, cols=2)\
    #     .add_traces([go.Scatter(x=X, y=Y, mode='markers', marker=dict(color='blue', size=3), name=r'Sample')],
    #                 rows=[1], cols=[1]) \
    #     .add_traces([go.Scatter(x=X, y=[mu] * X.shape[0], mode='lines', marker=dict(color="black"),
    #                             name=r'$\widehat\mu$')], rows=[1], cols=[1]) \
    #     .add_traces([go.Scatter(x=X, y=[10] * X.shape[0], mode='lines', marker=dict(color="red"), name=r'$\mu$')],
    #                 rows=[1], cols=[1]) \
    #     .update_layout(title_text=r"$\text{(1) Generating Data From Probabilistic Model}$", height=300) \
    #     .show()

    # Question 2 - Empirically showing sample mean is consistent
    mus = []
    sigmas = []
    sample_sizes = np.arange(10, 1001, 10)
    for i in sample_sizes:
        Z = Z.fit(Y[0:i])
        mus.append(Z.mu_)
    mus = np.array(mus)
    make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=sample_sizes, y=np.absolute(mus - requested_mu), mode='lines',
                                marker=dict(color='blue'), name=r'$|\widehat\mu-\mu|$')],
                    rows=[1], cols=[1]) \
        .update_layout(title_text=r"$\text{(1) Error of expectancy}$", height=300) \
        .update_yaxes(title_text=r'$|\widehat\mu-\mu|$') \
        .update_xaxes(title_text="Sample Size") \
        .show()

    # Question 3 - Plotting Empirical PDF of fitted model
    make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=Y, y=Z.pdf(Y), mode='markers',
                                marker=dict(color='blue'), name=r'$PDF of samples$')],
                    rows=[1], cols=[1]) \
        .update_layout(title_text=r"$\text{emperical PDF plot of samples of }\mathcal{N}\left(0,1\right)$", height=300) \
        .update_yaxes(title_text=r'$PDF$') \
        .update_xaxes(title_text="Values of Sample") \
        .show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
