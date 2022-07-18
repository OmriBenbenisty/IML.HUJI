import numpy as np
import pandas as pd

from IMLearn.base.base_module import BaseModule
from IMLearn.metrics.loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network
    Attributes:
    -----------
    input_dim_: int
        Size of input to layer (number of neurons in preceding layer
    output_dim_: int
        Size of layer output (number of neurons in layer_)
    activation_: BaseModule
        Activation function to be performed after integration of inputs and weights
    weights: ndarray of shape (input_dim_, outout_din_)
        Parameters of function with respect to which the function is optimized.
    include_intercept: bool
        Should layer include an intercept or not
    """

    def __init__(self, input_dim: int, output_dim: int, activation: BaseModule = None, include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer
        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer
        output_dim: int
            Size of layer output (number of neurons in layer_)
        activation_: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer
        include_intercept: bool, default=True
            Should layer include an intercept or not
        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """
        super().__init__()
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.activation_ = activation
        self.include_intercept_ = include_intercept
        mu = 0
        input_dim = input_dim + 1 if include_intercept else input_dim
        sigma = 1 / input_dim
        self.weights_ = np.random.normal(mu, sigma, size=(input_dim, output_dim))

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x: output value of layer at point
        self.weights and given input
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights
        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self.weights
        """
        _X = np.c_[(X, np.ones((X.shape[0], 1)))] if self.include_intercept_ else X
        return self.activation_.compute_output(X=_X @ self.weights_) if self.activation_ \
            else _X @ self.weights_

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights
        Returns:
        -------
        output: ndarray of shape (input_dim, n_samples)
            Derivative with respect to self.weights at point self.weights
        """
        _X = np.c_[X, np.ones(X.shape[0])] if self.include_intercept_ else X
        return self.activation_.compute_jacobian(X=_X @ self.weights_) if self.activation_ \
            else _X
        # return X.T
        # _X = np.hstack((X, np.ones(X.shape[0]))) if self.include_intercept_ else X
        # return self.activation_.compute_jacobian(X=self.weights_ @ _X, **kwargs)


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation
        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        return np.maximum(X, 0)

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to given data
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to compute derivative with respect to
        Returns:
        -------
        output: ndarray of shape (n_samples,)
            Element-wise derivative of ReLU with respect to given data
        """
        return np.sign(self.compute_output(X))


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """

    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the Cross-Entropy over the Softmax of given data, with respect to every
        CrossEntropy(Softmax(x),e_k) for every sample x
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data for which to compute the cross entropy loss
        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed
        Returns:
        --------
        output: ndarray of shape (n_samples,)
            cross-entropy loss value of given X and y
        """
        n_samples, input_dim = X.shape

        return cross_entropy(y_true=y, y_pred=softmax(X))
        # def c_e(_X):
        #     y_true = np.zeros(input_dim)
        #     _y = int(_X[input_dim])
        #     y_true[_y] = 1
        #     return cross_entropy(y_true, softmax(_X[0:input_dim]))
        #
        # return np.apply_along_axis(c_e, axis=1, arr=np.c_[X, y])

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the derivative of the cross-entropy loss function with respect to every given sample
        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data with respect to which to compute derivative of the cross entropy loss
        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed
        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            derivative of cross-entropy loss with respect to given input
        """
        y_dummy = pd.get_dummies(y).to_numpy()
        s_max = softmax(X)
        ret = - y_dummy / s_max
        assert ret.shape == X.shape
        return ret

        # def jacobian_softmax(s):
        #     return np.diag(s) - np.outer(s, s)


        # s_max = softmax(X)
        # # b does one-hot encoding of y
        # b = np.zeros_like(s_max)
        # b[np.arange(s_max.shape[0]), y] = 1
        #
        # return s_max - b
        # s_max = softmax(X)
        # return - y / s_max
        # return np.array([jacobian_softmax(row) for row in s])
        # return np.diag(s) - np.outer(s, s)
