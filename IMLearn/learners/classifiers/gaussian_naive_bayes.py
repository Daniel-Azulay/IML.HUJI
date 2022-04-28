from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # get each class and the number of appearences n_k :
        self.classes_, classes_counts = np.unique(y, return_counts=True)
        # according to theoretical part pi_k is deviding n_k by m , so we get:
        self.pi_ = classes_counts / y.shape[0]

        # compute mu_ array for each class:
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i, cur_class in enumerate(self.classes_):
            # creating the whole array of this class:
            self.mu_[i] = np.mean(X[np.array(np.where(y == cur_class))[0]], axis=0)

        # compute vars matrix:
        # matrix is from shape(K,d)
        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for k in self.classes_:
            self.vars_[k] = np.var(X[np.array(np.where(y == k))[0]], axis=0)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        # Calculating the likelihood for each of the classes
        for l, k in enumerate(self.classes_):
            cov_k_matrix_inverse = np.diag(1/self.vars_[l])
            cov_k_matrix_det = np.prod(self.vars_[l])
            Z = np.sqrt(((2*np.pi)**X.shape[0]) * cov_k_matrix_det)
            mahalanobis = np.einsum("bi,ij,bj->b", X - self.mu_[k, :], cov_k_matrix_inverse, X - self.mu_[k, :])

            likelihoods[:, k] = self.pi_[k] * np.exp(-.5 * mahalanobis) / Z

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
