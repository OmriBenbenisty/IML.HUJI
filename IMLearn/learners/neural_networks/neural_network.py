import numpy as np
from typing import List, Union, NoReturn

import pandas as pd

from IMLearn.base.base_module import BaseModule
from IMLearn.base.base_estimator import BaseEstimator
from IMLearn.desent_methods import StochasticGradientDescent, GradientDescent
from .modules import FullyConnectedLayer


class NeuralNetwork(BaseEstimator, BaseModule):
    """
    Class representing a feed-forward fully-connected neural network
    Attributes:
    ----------
    modules_: List[FullyConnectedLayer]
        A list of network layers, each a fully connected layer with its specified activation function
    loss_fn_: BaseModule
        Network's loss function to optimize weights with respect to
    solver_: Union[StochasticGradientDescent, GradientDescent]
        Instance of optimization algorithm used to optimize network
    pre_activations_:
    """
    pre_activations_: List
    post_activations_: List

    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()
        self.modules_ = modules
        self.loss_fn_ = loss_fn
        self.solver_ = solver

    # region BaseEstimator implementations
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit network over given input data using specified architecture and solver
        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # self.compute_output(X, y)
        # self.compute_jacobian(X,y)
        # self.solver_.fit(f=self, X=X, y=pd.get_dummies(y).to_numpy())
        self.solver_.fit(f=self, X=X, y=y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given samples using fitted network
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted labels of given samples
        """
        preds = self.compute_prediction(X=X)
        # return np.apply_along_axis(np.max, axis=1, arr=preds)
        return np.argmax(preds, axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates network's loss over given data
        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        --------
        loss : float
            Performance under specified loss function
        """
        loss = self.loss_fn_.compute_output(X=self.predict(X), y=y)
        return float(np.mean(loss))

    # endregion

    # region BaseModule implementations
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network output with respect to modules' weights given input samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        output: ndarray of shape (1,)
            Network's output value including pass through the specified loss function
        Notes
        -----
        Function stores all intermediate values in the `self.pre_activations_` and `self.post_activations_` arrays
        """
        pred = self.compute_prediction(X)
        return self.loss_fn_.compute_output(X=pred, y=y, **kwargs)

        # pre_i = self.loss_fn_.compute_output(X=post_i @ self.loss_fn_.weights)
        # self.pre_activations_.append(pre_i)
        #
        # post_i = module.compute_output(pre_i)
        # self.post_activations_.append(post_i)
        # self.post_activations_.append()

    def compute_prediction(self, X: np.ndarray):
        """
        Compute network output (forward pass) with respect to modules' weights given input samples, except pass
        through specified loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        Returns
        -------
        output : ndarray of shape (n_samples, n_classes)
            Network's output values prior to the call of the loss function
        """
        post_i = X
        self.pre_activations_ = [0]
        self.post_activations_ = [post_i]

        # Forward Pass
        for module in self.modules_:
            temp = np.c_[post_i, np.ones(post_i.shape[0])] if module.include_intercept_ else post_i
            pre_i = temp @ module.weights
            self.pre_activations_.append(pre_i)

            post_i = module.compute_output(post_i)
            self.post_activations_.append(post_i)

        # loss = self.loss_fn_.compute_output(X=self.post_activations_[-1] - y)
        # self.post_activations_.append(loss)
        return self.post_activations_[-1]

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network's derivative (backward pass) according to the backpropagation algorithm.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        A flattened array containing the gradients of every learned layer.
        Notes
        -----
        Function depends on values calculated in forward pass and stored in
        `self.pre_activations_` and `self.post_activations_`
        """
        # Backward Pass
        n_samples = X.shape[0]

        n_layers = len(self.modules_)

        # derivative_chain = []
        # partial_der = []
        # delta = self.post_activations_[-1] - y  # loss

        delta_t = self.loss_fn_.compute_jacobian(X=self.post_activations_[-1], y=y, **kwargs)

        partial_der = []
        o_t_1 = self.post_activations_[-1]
        # partial_der = np.empty(len(self.modules_), dtype=object)
        for i in range(n_layers):
            module = self.modules_[n_layers - i - 1]
            w_t_1 = module.weights
            o_t = self.post_activations_[n_layers - i - 1]
            jac = module.compute_jacobian(o_t)

            # if module.include_intercept_:
            #     weights = np.delete(weights, 0, axis=1)
            #     activation = np.c_[activation, np.ones(activation.shape[0])]
            del_jac = np.einsum('ij,ik->jk', delta_t, jac)
            partial_der.append(o_t_1 @ del_jac)
            delta_t = (del_jac @ w_t_1.T)
            o_t_1 = o_t

        partial_der.reverse()

        # for t, module in enumerate(reversed(self.modules_), start =1):
        #     if module.activation_:
        #         jac = module.activation_.compute_jacobian(X=self.pre_activations_[n_layers - t - 1])
        #     else:
        #         shape = self.pre_activations_[n_layers - t - 1].shape
        #         jac = np.ones(shape=(shape[0], shape[1] + 1)) if module.include_intercept_ else np.ones(shape)
        #         # jac = np.einsum('ij,kj->ikj', jac, np.eye(jac.shape[1], dtype=jac.dtype))
        #
        #     # if module.include_intercept_:
        #     #     partial_der[n_layers - t - 1] =
        #     # else:
        #     partial_der[n_layers - t - 1] = self.post_activations_[n_layers - t - 1].T @ (delta * jac) / n_samples
        #     delta = (delta * jac) @ module.weights.T

        # derivative_chain.append(delta)
        # partial_der.append(np.dot(delta, self.post_activations_[-2].transpose()))
        #
        # last = n_layers - 2
        # for i in range(last, 0, -1):
        #     z = self.pre_activations_[i]
        #     derivative = self.modules_[i].compute_jacobian(z)
        #     delta = np.dot(self.modules_[i+1].weights_.transpose(), delta) * derivative
        #     # derivative_chain.append(delta)
        #     partial_der.append(np.dot(delta, self.post_activations_[i-1].transpose()))

        # derivative_chain.reverse()
        # partial_der.reverse()
        for i in range(len(partial_der)):
            partial_der[i] = np.mean(partial_der[i], axis=0).T

        return self._flatten_parameters(params=partial_der)

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        # deltas[last] = self.post_activations_[-1]
        #
        # # Compute gradient for the last layer
        # self._compute_loss_grad(
        #     last, n_samples, activations, deltas, coef_grads, intercept_grads
        # )
        #
        # inplace_derivative = DERIVATIVES[self.activation]
        # # Iterate over the hidden layers
        # for i in range(last, 0, -1):
        #     deltas[i - 1] = np.dot(deltas[i], self.pre_activations_[i].T)
        #     self.modules_[i].compute_jacobian(activations[i], deltas[i - 1])
        #
        #     self._compute_loss_grad(
        #         i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
        #     )
        #
        # return loss, coef_grads, intercept_grads

    @property
    def weights(self) -> np.ndarray:
        """
        Get flattened weights vector. Solvers expect weights as a flattened vector
        Returns
        --------
        weights : ndarray of shape (n_features,)
            The network's weights as a flattened vector
        """
        return NeuralNetwork._flatten_parameters([module.weights for module in self.modules_])

    @weights.setter
    def weights(self, weights) -> None:
        """
        Updates network's weights given a *flat* vector of weights. Solvers are expected to update
        weights based on their flattened representation. Function first un-flattens weights and then
        performs weights' updates throughout the network layers
        Parameters
        -----------
        weights : np.ndarray of shape (n_features,)
            A flat vector of weights to update the model
        """
        non_flat_weights = NeuralNetwork._unflatten_parameters(weights, self.modules_)
        for module, weights in zip(self.modules_, non_flat_weights):
            module.weights = weights

    # endregion

    # region Internal methods
    @staticmethod
    def _flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
        """
        Flattens list of all given weights to a single one dimensional vector. To be used when passing
        weights to the solver
        Parameters
        ----------
        params : List[np.ndarray]
            List of differently shaped weight matrices
        Returns
        -------
        weights: ndarray
            A flattened array containing all weights
        """
        return np.concatenate([grad.flatten() for grad in params])

    @staticmethod
    def _unflatten_parameters(flat_params: np.ndarray, modules: List[BaseModule]) -> List[np.ndarray]:
        """
        Performing the inverse operation of "flatten_parameters"
        Parameters
        ----------
        flat_params : ndarray of shape (n_weights,)
            A flat vector containing all weights
        modules : List[BaseModule]
            List of network layers to be used for specifying shapes of weight matrices
        Returns
        -------
        weights: List[ndarray]
            A list where each item contains the weights of the corresponding layer of the network, shaped
            as expected by layer's module
        """
        low, param_list = 0, []
        for module in modules:
            r, c = module.shape
            high = low + r * c
            param_list.append(flat_params[low: high].reshape(module.shape))
            low = high
        return param_list
    # endregion

"""



"""


