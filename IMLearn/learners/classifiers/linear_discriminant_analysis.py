from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # get each class and the number of appearences n_k :
        self.classes_ , classes_counts = np.unique(y, return_counts=True)
        # according to theoretical part pi_k is deviding n_k by m , so we get:
        self.pi_ = classes_counts / y.shape[0]

        # compute mu_ array for each class:
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i, cur_class in enumerate(self.classes_):
            # creating the whole array of this class:
            self.mu_[i] = np.mean(X[np.array(np.where(y == cur_class))[0]], axis=0)

        # compute cov_ matrix:
        # matrix is from shape(d,d)
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for i in self.classes_:
            xi_minus_mu_i = X[np.array(np.where(y == i))[0]]-self.mu_[i]
            # xi_minus_mu_i is a row vector instead of a column - meaning its already the transpose.
            self.cov_ = self.cov_ + np.transpose(xi_minus_mu_i) @ xi_minus_mu_i
        self.cov_ = self.cov_ / (X.shape[0] - self.classes_.shape[0]) # cov is the sum devided by m-K

        self._cov_inv = np.linalg.inv(self.cov_)
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

        likelihoods = np.zeros((X.shape[0], len(self.classes_)))
        for i, x in enumerate(X): # running through samples
            for k in self.classes_:
                likelihoods[i][k] = np.log(self.pi_)[k] + x @ self._cov_inv @ self.mu_[k] \
                                          - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]

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
