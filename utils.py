"""Collection of utility functions."""


import numpy as np


def sigmoid(x):
    """Compute sigmoid element-wise.

    The sigmoid of a scalar x is 1 / (1 + e^{-x}).

    Parameter
    ---------
    x: ndarray or scalar
        input
    Returns
    -------
    ndarray or scalar, same shape of input x
        element-wise sigmoid of the input
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """Compute the gradient of sigmoid element-wise.

    The gradient of sigmoid at x equals sigmoid(x)*(1 - sigmoid (x)).

    Parameter
    ---------
    x: ndarray or scalar
        input
    Returns
    -------
    ndarray or scalar, same shape of input x
        element-wise gradient of sigmoid of the input
    """
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def softmax(x, axis=None):
    """Compute the softmax of array elements along a given axis.

    Parameter
    ---------
    x: ndarray
        input array
    axis: None or int or tuple of ints, optional
        axis or axes along which the softmax is performed. The default,
        axis=None, will perform the softmax over all elements. If axis
        is negative it counts from the last to the first axis. If axis
        is a tuple of ints, the softmax is performed on all of the axes
        specified in tuple.

    Returns
    -------
    ndarray, same shape of input x
        softmax of the input along axis
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
