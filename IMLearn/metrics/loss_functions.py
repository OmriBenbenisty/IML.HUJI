import numpy as np
import pandas as pd


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.square(y_true - y_pred).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    return np.sum(y_true != y_pred) / y_true.shape[0] if normalize \
        else np.sum(y_true != y_pred)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    return ((y_true - y_pred) == 0).astype(int).mean()


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 5000)
    return -np.sum(y_true * np.log(y_pred))



def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data
    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)
    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    X_ = np.clip(X, -709.78, 709.78)
    e_X = np.exp(X_)

    return e_X / np.sum(e_X, axis=1,keepdims=True)

