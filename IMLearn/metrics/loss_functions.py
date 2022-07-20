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
    # prod = y_true * y_pred
    # return np.sum((prod < 0).astype(int)) / prod.shape[0] if normalize\
    #     else np.sum((prod < 0).astype(int))

    # return np.linalg.norm(y_true - y_pred) / y_true.shape[0] if normalize \
    #     else np.linalg.norm(y_true - y_pred)


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
    # y_pred =  np.argmax(y_pred, axis=1)

    return -np.sum(y_true * np.log(y_pred))

    # if y_prob.shape[1] == 1:
    #     y_prob = np.append(1 - y_prob, y_prob, axis=1)
    #
    # if y_true.shape[1] == 1:
    #     y_true = np.append(1 - y_true, y_true, axis=1)

    b = pd.get_dummies(y_true).to_numpy()

    # b = np.zeros_like(y_pred)
    # b[np.arange(y_pred.shape[0]), y_true] = 1

    return -np.sum(b * np.log(y_pred), axis=1)
    # return -(y_true * np.log(y_prob)).sum() / y_prob.shape[0]


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
    X = np.clip(X, -709.78, 709.78)
    e_X = np.exp(X)
    # e_X = np.clip(e_X, 1e-10, np.inf)

    return e_X / np.sum(e_X, axis=1,keepdims=True)

    return (e_X.T / np.sum(e_X, axis=1)).T
    sum_X = np.sum(e_X, axis=1, keepdims=True)
    ret = e_X / sum_X
    assert e_X.shape[0] == sum_X.shape[0]
    assert ret.shape == X.shape
    return ret
    # e_X = np.exp(X)
    # sum_X = np.sum(e_X) if X.ndim == 1 else np.sum(e_X, axis=1)
    # return  e_X / sum_X if X.ndim == 1 else np.apply_along_axis(lambda x: x / sum_X, axis=0, arr=e_X)
