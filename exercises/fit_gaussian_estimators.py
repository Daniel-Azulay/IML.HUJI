from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from utils import *

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    requested_mu = 10
    requested_sigma = 1
    Y = np.random.normal(requested_mu, requested_sigma, size=1000)
    Z = UnivariateGaussian().fit(Y)
    mu_hat, sigma_hat = Z.mu_, Z.var_
    print(mu_hat, sigma_hat)

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


def cartesian_product_rows(vec1, vec2):
    # np.repeat([1, 2, 3], 4) -> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    # np.tile([1, 2, 3], 4)   -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    return np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))])


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    Y = np.random.multivariate_normal(mu, sigma, size=1000)
    Z = MultivariateGaussian().fit(Y)
    mu_hat, sigma_hat = Z.mu_, Z.cov_
    print(mu_hat)
    print(sigma_hat)
    # Question 5 - Likelihood evaluation
    mu_one = np.linspace(-10, 10, 200)
    mu_three = np.linspace(-10, 10, 200)
    cart_prod = cartesian_product_rows(mu_one, mu_three)
    all_mus = np.transpose(np.array([cart_prod[:1, :][0], [0] * (200*200), cart_prod[1:, :][0], [0] * (200*200)]))
    print(all_mus)
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    #raise NotImplementedError()




if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
