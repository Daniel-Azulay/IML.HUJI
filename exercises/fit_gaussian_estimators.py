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
        .update_layout(title_text=r"$\text{(1) Error of expectancy}$", height=300, width=800) \
        .update_yaxes(title_text=r'$|\widehat\mu-\mu|$') \
        .update_xaxes(title_text="Sample Size") \
        .show()

    # Question 3 - Plotting Empirical PDF of fitted model
    make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=Y, y=Z.pdf(Y), mode='markers',
                                marker=dict(color='blue'), name=r'$PDF of samples$')],
                    rows=[1], cols=[1]) \
        .update_layout(title_text=r"$\text{emperical PDF plot of samples of }\mathcal{N}\left(0,1\right)$",
                       height=300, width=800) \
        .update_yaxes(title_text=r'$PDF$') \
        .update_xaxes(title_text="Values of Sample") \
        .show()


# def cartesian_product_rows(vec1, vec2):
#     # np.repeat([1, 2, 3], 4) -> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
#     # np.tile([1, 2, 3], 4)   -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
#     return np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))])


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    mu_samples_number = 200

    multivariate_sample = np.random.multivariate_normal(mu, sigma, size=1000)
    fitted_multivariate_gaussian = MultivariateGaussian().fit(multivariate_sample)
    mu_hat, sigma_hat = fitted_multivariate_gaussian.mu_, fitted_multivariate_gaussian.cov_
    print(mu_hat)
    print(sigma_hat)

    # Question 5 - Likelihood evaluation
    mus_data = np.linspace(-10, 10, mu_samples_number)
    log_likelihoods = np.zeros((mu_samples_number, mu_samples_number))
    for i in range(mu_samples_number):
        for j in range(mu_samples_number):
            cur_mu = np.array([mus_data[i], 0, mus_data[j], 0])
            log_likelihoods[i][j] = fitted_multivariate_gaussian.log_likelihood(cur_mu, sigma, multivariate_sample)

    go.Figure() \
        .add_trace(go.Heatmap(x=mus_data, y=mus_data, z=log_likelihoods, colorscale='Blues')) \
        .update_yaxes(title_text=r'$f_1\text{ value}$') \
        .update_xaxes(title_text=r"$f_3\text{ value}$") \
        .update_layout(title=r"$\text{Heatmap of log-likelihood as a function of }f_1, f_3$", height=600, width=1000) \
        .show()

    # Question 6 - Maximum likelihood
    maximum_row = np.argmax(log_likelihoods, axis=0)
    maximum_col = np.argmax(log_likelihoods, axis=1)
    print(round(mus_data[maximum_row[0]], 3), round(mus_data[maximum_col[0]], 3))

    # quiz_array = [1, 5, 2,  3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #               -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2]
    # quiz_gaussian = UnivariateGaussian().fit(np.array(quiz_array))
    # print("log likelihood for quiz is: ", quiz_gaussian.log_likelihood(10, 1, np.array(quiz_array)))
if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
