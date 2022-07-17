import numpy as np


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
    eps = 1e-15
    y_prob = np.clip(y_pred, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -(y_true * np.log(y_prob)).sum() / y_prob.shape[0]


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
    e_X = np.exp(X)
    sum_X = np.sum(e_X, axis=0)
    return np.apply_along_axis(lambda x: x / sum_X, axis=0, arr=X)
