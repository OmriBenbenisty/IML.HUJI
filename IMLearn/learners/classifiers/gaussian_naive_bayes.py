from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


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
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        n_samples = y.shape[0]
        self.mu_ = np.ndarray(shape=(n_classes, n_features))
        self.vars_ = np.ndarray(shape=(n_classes, n_features))
        self.pi_ = np.ndarray(n_classes)
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.mu_[i] = np.mean(X_k, axis=0)
            self.pi_[i] = X_k.shape[0] / n_samples
            self.vars_[i] = np.var(X_k, axis=0)

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
        pred = np.empty((self.classes_.shape[0], X.shape[0]))  # (n_classes, n_samples)

        for i in range(len(self.classes_)):
            mu_k = self.mu_[i]  # (n_features)
            log_pi_k = np.log(self.pi_[i])  # (1)
            sigma_k = self.vars_[i]  # (n_features)
            log_sigma_k = np.log(sigma_k)  # (n_features)
            sum_X = np.sum(
                (((X - mu_k) ** 2) / sigma_k) + log_sigma_k, axis=1
            )  # (n_samples)
            pred[i] = log_pi_k - 0.5 * sum_X  # (1) - (n_samples) -> (n_samples)
        # pred = np.argmax(pred.T, axis=1)
        pred = np.argmax(pred, axis=0)  # (n_samples)
        return np.fromiter(map(lambda y: self.classes_[y], pred),
                           dtype=type(self.classes_[0]))  # (n_samples)

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
        prob = np.empty((self.classes_.shape[0], X.shape[0]))  # (n_classes, n_samples)
        for i in range(len(self.classes_)):
            mu_k = self.mu_[i]  # (n_features)
            var_k = self.vars_[i]  # (n_features)
            pi_k = self.pi_[i]  # (1)
            X_mu_k = X - mu_k  # (n_samples, n_features) - (n_features) -> (n_samples, n_features)
            prob_X = np.prod((np.exp(-0.5 * (X_mu_k ** 2))
                              / np.sqrt((2 * np.pi) * var_k)), axis=1)  # (n_samples, n_features) -> (n_samples)
            prob[i] = (pi_k * prob_X)  # (n_samples)

        return prob.T  # (n_samples, n_classes)

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
