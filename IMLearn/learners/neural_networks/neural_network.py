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
    pre_activations_: np.ndarray
    post_activations_: np.ndarray

    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()
        self.modules_ = modules
        self.loss_fn_ = loss_fn
        self.solver_ = solver
        self.pre_activations_ = np.empty(len(modules) + 1, dtype=object)
        self.post_activations_ = np.empty(len(modules) + 1, dtype=object)

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
        print(loss.shape)
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

        o = X
        self.pre_activations_[0] = 0
        # self.post_activations_[0] = np.c_[X, np.ones(X.shape[0])] if self.modules_[0].include_intercept_ else X
        self.post_activations_[0] = o

        for t, layer in enumerate(self.modules_):
            temp = np.c_[o, np.ones(o.shape[0])] if layer.include_intercept_ else o
            weights = np.r_[np.atleast_2d(layer.bias_), layer.weights_] if layer.include_intercept_ else layer.weights_

            a = temp @ weights
            self.pre_activations_[t + 1] = a
            if layer.activation_:
                o = layer.activation_.compute_output(X=a)
            else:
                o = a
            self.post_activations_[t + 1] = o

        return self.post_activations_[-1]

        post_i = X.copy()
        self.pre_activations_ = []
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
        ret = self.post_activations_[-1]
        assert ret.shape[0] == X.shape[0]
        assert ret.shape[1] == self.modules_[-1].output_dim_
        return ret

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

        n_layers = len(self.modules_)
        delta_t = self.loss_fn_.compute_jacobian(X=self.post_activations_[-1], y=y, **kwargs)

        partials = np.empty(len(self.modules_), dtype=object)
        for t, layer in enumerate(reversed(self.modules_)):
            if layer.activation_:
                jac = layer.activation_.compute_jacobian(X=self.post_activations_[n_layers - t])
            else:
                jac = np.ones_like(self.post_activations_[n_layers - t])
            partials[n_layers - t - 1] = self.post_activations_[n_layers - t - 1].T @ (delta_t * jac) / len(X)
            delta_t = (delta_t * jac) @ layer.weights.T
        ret = self._flatten_parameters(partials)
        assert ret.shape[0] == np.sum([f.weights.size for f in self.modules_])
        return ret


        # partials = []  # array to store derivatives per layer
        #
        # # initialize delta value (used to save up calculations in backpropagation)
        # delta_T = self.loss_fn_.compute_jacobian(X=self.post_activations_[-1], y=y)
        # delta_T = np.einsum('kjl,kl->kj', self.modules_[-1].compute_jacobian(X=self.pre_activations_[-1]), delta_T)
        #
        # # backpropagate from last layer to first layer, note the index i starts from 1
        # for i, module in enumerate(reversed(self.modules_), start=1):
        #     # calculate the derivative of objective with respect to the weights of the current layer
        #     if module.include_intercept_:  # add bias term
        #         final_partial = np.einsum('ki,kj->kij', delta_T, np.concatenate((np.ones((self.post_activations_[-i-1].shape[0], 1)), self.post_activations_[-i - 1]), axis=1))
        #     else:
        #         final_partial = np.einsum('ki,kj->kij', delta_T, self.post_activations_[-i - 1])
        #
        #     partials.append(final_partial)  # save the derivative
        #
        #     # update delta value
        #
        #     if i < len(self.modules_):  # in the last iter we don't update delta (+we can't, index error)
        #         activation_derivative = self.modules_[-i-1].activation_.compute_jacobian(X=self.pre_activations_[-i-1])
        #
        #         if module.include_intercept_:  # if there's a bias, don't use it since it doesn't affect previous layers
        #             weights_times_delta = np.einsum('il,kl->ki', module.weights[1:, :], delta_T)
        #             delta_T = np.einsum('kil,kl->ki', activation_derivative, weights_times_delta)
        #         else:
        #             weights_times_delta = np.einsum('il,kl->ki', module.weights, delta_T)
        #             delta_T = np.einsum('kil,kl->ki', activation_derivative, weights_times_delta)
        #
        # # reverse partials to get correct order
        # partials = partials[::-1]
        #
        # # get the average of the derivatives and transpose for correct shape
        # for i in range(len(partials)):
        #     partials[i] = np.mean(partials[i], axis=0).T
        #
        # # return the flattened array
        # return self._flatten_parameters(partials)

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
            del_jac = delta_t * jac
            # if module.include_intercept_:
            #     weights = np.delete(weights, 0, axis=1)
            #     activation = np.c_[activation, np.ones(activation.shape[0])]
            # del_jac = np.einsum('ij,ik->jk', delta_t, jac)
            # del_jac = jac @ delta_t
            # der = np.einsum('kj,jj->kjj', o_t_1, del_jac)
            der =o_t_1.T @ del_jac
            # partial_der.append(o_t_1 @ del_jac)
            partial_der.append(der)
            delta_t = (del_jac @ w_t_1.T)
            o_t_1 = o_t

        # ret = self._flatten_parameters(params=partial_der)
        # # assert ret.shape[0] == n_samples
        # assert ret.shape[0] == np.sum([f.weights.size for f in self.modules_])
        # return ret
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

        ret = self._flatten_parameters(params=partial_der)
        # assert ret.shape[0] == n_samples
        assert ret.shape[0] == np.sum([f.weights.size for f in self.modules_])
        return ret



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


