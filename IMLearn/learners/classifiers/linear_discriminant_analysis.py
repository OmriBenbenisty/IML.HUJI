from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


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
        # _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        # self.priors_ = np.bincount(y_t) / float(len(y))

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        n_samples = y.shape[0]
        self.mu_ = np.zeros(shape=(n_classes, n_features))
        self.pi_ = np.zeros(n_classes)
        self.cov_ = np.zeros((n_features, n_features))
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.mu_[i] = np.mean(X_k, axis=0)  # np.sum(X_k) / counts[i]
            self.pi_[i] = X_k.shape[0] / n_samples
            X_k_mu_yi = X_k - self.mu_[i]  # n_k,n_features
            self.cov_ += X_k_mu_yi.T.dot(X_k_mu_yi)  # n_features,n_k  * n_k,n_features -> (n_features, n_features)
        self.cov_ /= (n_samples - n_classes)  # divide by  m-K
        self._cov_inv = inv(self.cov_)  # (n_features, n_features)
        self._a = (self._cov_inv @ self.mu_.T).T
        # ((n_features,n_features) * (n_features,n_classes)).T ->
        # (n_features,n_classes).T -> (n_classes, n_features)
        self._b = (-0.5 * np.einsum('ij,ij->i', self.mu_ @ self._cov_inv, self.mu_) +
                   np.log(self.pi_))  # ->  (n_classes) + (n_classes) -> (n_classes)
        # - (n_classes, n_features) @ (n_features, n_features) *(dot) (n_classes, n_features) + (n_classes)

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

        # coef_ = np.linalg.lstsq(self.cov_, self.mu_.T)[0].T
        # intercept_ = -0.5 * np.diag(np.dot(self.mu_, coef_.T)) + np.log(
        #     self.priors_)
        # pred = np.zeros(X.shape[0])
        # for i, c in enumerate(self.classes_):
        #     pred[i] = np.dot(self.mu_[i] @ self.cov_, X) - \
        #               np.dot(self.mu_[i] @ self.cov_, self.mu_[i])*0.5 +\
        #               np.log(self.pi_[i])
        # return pred
        pred = np.empty((self.classes_.shape[0], X.shape[0]))  # (n_classes, n_samples)
        #  self._a @ X + self._b    (n_features,n_classes) * (n_samples, n_features) + (n_classes, n_classes) ->
        #  self._a.T @ X.T + self._b    (n_classes,n_features) * (n_features,n_samples) + (n_classes, n_classes) ->
        for i in range(len(self.classes_)):
            a_k = self._a[i]  # (n_features)
            b_k = self._b[i]  # (1)
            pred[i] = a_k @ X.T + b_k  # (1,n_features) * (n_features, n_samples) + (1) -> (n_samples)
        # pred = np.argmax(pred.T, axis=1)
        pred = np.argmax(pred, axis=0)  # (n_samples)
        return np.fromiter(map(lambda y: self.classes_[y], pred),
                           dtype=type(self.classes_[0]))  # (n_samples)

        # return np.fromiter(map(lambda y: self.classes_[y],
        #                        np.argmax(self._a @ X + self._b, axis=1)),
        #                    dtype=type(self.classes_[0]))

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
        #  self._a @ X + self._b    (n_features,n_classes) * (n_samples, n_features) + (n_classes, n_classes) ->
        #  self._a.T @ X.T + self._b    (n_classes,n_features) * (n_features,n_samples) + (n_classes, n_classes) ->
        for i in range(len(self.classes_)):
            mu_k = self.mu_[i]  # (n_features)
            det_cov = det(self.cov_)
            pi_k = self.pi_[i]
            X_mu_k = X - mu_k  # (n_samples, n_features) - (n_features) -> (n_samples, n_features)
            prob[i] = (pi_k *
                       (np.exp(-0.5 * np.einsum('ij,ji->i', X_mu_k @ self._cov_inv, X_mu_k.T))
                        / np.sqrt(((2 * np.pi) ** X.shape[1]) * det_cov)))
            # (n_samples,n_features) @  (n_features, n_features) * (n_samples, n_features)-> (n_samples)
            #  (n_features) * (n_features, n_samples) + (n_classes)

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
