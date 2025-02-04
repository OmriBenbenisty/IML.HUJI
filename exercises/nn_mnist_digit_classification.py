import time
import numpy as np
import gzip
from typing import Tuple, Callable, List

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = 'browser'


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset
    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set
    train_y : ndarray of shape (60,000,)
        Responses of training samples
    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set
    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images
    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid
    title : str, default="
        Title to add to figure
    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    values = []
    weights = []
    gradients = []

    def callback(**kwargs) -> None:
        values.append(kwargs['val'])
        gradients.append(kwargs['grad'])
        if kwargs["t"] % 100 == 0:
            weights.append(kwargs['weights'])

    return callback, values, weights, gradients


def get_gd_time_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    values = []
    times = []

    def callback(**kwargs) -> None:
        values.append(kwargs['val'])
        times.append(time.time())

    return callback, values, times


def get_sgd_time_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    values = []
    times = []

    def callback(**kwargs) -> None:
        values.append(kwargs['val'])
        times.append(time.time())

    return callback, values, times


if __name__ == '__main__':
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10
    width = 64

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network

    callback, values, deltas, grads = get_gd_state_recorder_callback()

    nn = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=n_classes)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(
            learning_rate=FixedLR(base_lr=0.1),
            max_iter=10000,
            batch_size=256,
            callback=callback)
    ).fit(train_X, train_y)

    pred_y = nn.predict(test_X)
    acc = accuracy(y_true=test_y, y_pred=pred_y)
    print(f"NN MNIST With Hidden Layers Acc On Test = {acc}")

    # Plotting convergence process
    conv_proc = go.Figure(data=[go.Scatter(x=np.arange(len(values)),
                                           y=[np.mean(value) for value in values],
                                           # y=[cross_entropy(y_true=train_y, y_pred=val) for val in values],
                                           name="Loss")],
                          layout=go.Layout(
                              title="Convergence Rate for NN with<br>"
                                    f"2 hidden layers of width {width}",
                              xaxis=dict(title=r"$\text{Iteration}$"),
                              yaxis=dict(title=r"$\text{Objective}$")))
    # add norm of weights
    conv_proc.add_trace(go.Scatter(x=np.arange(len(values)),
                                   y=[np.linalg.norm(grad) for grad in grads],
                                   name="Grad Norm"))
    conv_proc.show()

    # Plotting test true- vs predicted confusion matrix
    conf_mat = confusion_matrix(test_y, pred_y)
    conf_mat_sorted = np.argsort(np.diag(conf_mat))
    least_conf = conf_mat_sorted[:3]
    most_conf = conf_mat_sorted[-2:][::-1]
    print(f"Most Confident 2 = {most_conf}")
    print(f"Least Confident 3 = {least_conf}")
    conf_mat_g = go.Figure(
        data=[go.Heatmap(
            z=conf_mat,
            x=np.arange(n_classes),
            y=np.arange(n_classes)
        )],
        layout=go.Layout(
            title=f"Confusion Matrix",
            xaxis=dict(title=f"True Value", dtick=1),
            yaxis=dict(title=f"Predicted Value", dtick=1)
        )
    )

    conf_mat_g.show()

    raise Exception

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#

    callback, values, deltas, grads = get_gd_state_recorder_callback()

    nn_no_hidden = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=n_classes)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(
            learning_rate=FixedLR(base_lr=0.1),
            max_iter=10000,
            batch_size=256,
            callback=callback)
    ).fit(train_X, train_y)
    pred_y = nn_no_hidden.predict(test_X)
    acc = accuracy(y_true=test_y, y_pred=pred_y)
    print(f"NN MNIST No Hidden Acc On Test = {acc}")


    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    prob = nn.compute_prediction(X=test_X)
    confidence = np.max(prob, axis=1)
    # print(confidence)
    most_conf = np.argmax(confidence)
    least_conf = np.argmin(confidence)
    print(f"Most Confident = {most_conf}")
    print(f"Least Confident = {least_conf}")

    plot_images_grid(test_X[most_conf].reshape(1, 784),
                     title="Most Confident").show()
    plot_images_grid(test_X[least_conf].reshape(1, 784),
                     title="Least Confident").show()

    test_X_7 = test_X[test_y == 7]
    test_y_7 = test_y[test_y == 7]

    prob_7 = nn.compute_prediction(X=test_X_7)
    confidence_7 = np.max(prob_7, axis=1)
    confidence_7 = np.argsort(confidence_7)

    plot_images_grid(test_X_7[confidence_7[-64:]].reshape(64, 784),
                     title="Most Confident").show()
    plot_images_grid(test_X_7[confidence_7[:64]].reshape(64, 784),
                     title="Least Confident").show()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs SGD Running times                                                         #
    # ---------------------------------------------------------------------------------------------#

    train_size = 2500
    train_X = train_X[:train_size].copy()
    train_y = train_y[:train_size].copy()
    scatters = []

    sgd_callback, sgd_values, sgd_time = get_sgd_time_recorder_callback()
    sgd_nn = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=n_classes)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(
            learning_rate=FixedLR(base_lr=0.1),
            max_iter=10000,
            batch_size=64,
            tol=1e-10,
            callback=sgd_callback
        )
    ).fit(train_X.copy(), train_y.copy())
    print("Done Fitting")
    sgd_time = np.array(sgd_time) - sgd_time[0]
    sgd_scatter = go.Scatter(
        x=sgd_time.copy(),
        y=[np.mean(value) for value in sgd_values]
    )
    scatters.append(sgd_scatter)
    go.Figure(
        data=[
            sgd_scatter
        ],
        layout=go.Layout(
            title=f"NN with SGD Loss as Function of Runtime",
            xaxis_title="Runtime",
            yaxis_title="Loss"
        )
    ).show()

    np.random.seed(0)

    gd_callback, gd_values, gd_time = get_gd_time_recorder_callback()
    gd_nn = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=width, activation=ReLU()),
            FullyConnectedLayer(input_dim=width, output_dim=n_classes)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(
            learning_rate=FixedLR(base_lr=0.1),
            max_iter=10000,
            tol=1e-10,
            callback=gd_callback
        )
    ).fit(train_X.copy(), train_y.copy())
    print("Done Fitting")
    gd_time = np.array(gd_time) - gd_time[0]
    gd_scatter = go.Scatter(
        x=gd_time.copy(),
        y=[np.mean(value) for value in gd_values]
    )
    scatters.append(gd_scatter)
    go.Figure(
        data=[
            gd_scatter
        ],
        layout=go.Layout(
            title=f"NN with GD Loss as Function of Runtime",
            xaxis_title="Runtime",
            yaxis_title="Loss"
        )
    ).show()

    # for i, solver in enumerate(solvers):
    #     solver.callback_ = gd_callback
    #     print("Fitting....")

    go.Figure(
        data=scatters,
        layout=go.Layout(
            title="NN Loss as Function of Runtime",
            xaxis_title="Runtime",
            yaxis_title="Loss"
        )
    ).show()

    print("------Done------")
