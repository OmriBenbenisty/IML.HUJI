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
        weight_size = input_dim + 1 if include_intercept else input_dim
        sigma = 1 / weight_size
        self.weights = np.random.normal(mu, sigma, (input_dim, output_dim))

        self.bias_ = None
        if self.include_intercept_:
            self.bias_ = np.random.normal(mu, sigma) * np.ones(output_dim)

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
        assert X.shape[1] == self.input_dim_
        _X = np.c_[(X, np.ones((X.shape[0], 1)))] if self.include_intercept_ else X
        weights = np.r_[self.weights_, np.atleast_2d(self.bias_)] if self.include_intercept_ else self.weights_

        ret = self.activation_.compute_output(X=_X @ weights) if self.activation_ \
            else _X @ weights
        assert ret.shape[0] == X.shape[0]
        assert ret.shape[1] == self.output_dim_
        return ret

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
        assert X.shape[1] == self.input_dim_
        _X = np.c_[X, np.ones(X.shape[0])] if self.include_intercept_ else X
        weights = np.r_[ self.weights_, np.atleast_2d(self.bias_)] if self.include_intercept_ else self.weights_
        if self.activation_:
            ret = self.activation_.compute_jacobian(X=_X @ weights) @ _X.T
        else:
            _Xw = _X @ weights
            derivative = np.ones_like(_Xw)
            ret = derivative @ _X.T
        return ret



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
        ret = np.maximum(X, 0)
        assert ret.shape == X.shape
        return ret

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
        ret = np.sign(self.compute_output(X))
        assert ret.shape[0] == X.shape[0]
        # assert ret.ndim == 1
        return ret


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
        softmax_result = -np.log(softmax(X))
        ret = softmax_result[np.arange(len(y)), y]
        assert ret.shape == y.shape
        return ret


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
        E_K = np.zeros(X.shape)
        for i in range(len(y)):
            # filling E to e_k for each sample x
            E_K[i, y[i]] = 1
        return softmax(X) - E_K
