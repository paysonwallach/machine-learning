# Module of activation functions for use in matrix-based neural networks

import numpy as np


def sigmoid(X, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + np.exp(-X))
    else:
        return sigmoid(X)*(1.0 - sigmoid(X))


def softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z


def relu(x, deriv=False):
    if not deriv:
        if x <= 0:
            return 0

    else:
        if x <= 0:
            return 0
        else:
            return 1


def prelu(X, alpha=0.1, deriv=False):
    if not deriv:
        for x in np.nditer(X, op_flags=['readwrite']):
            if x[...] <= 0:
                x[...] *= alpha

    else:
        for x in np.nditer(X, op_flags=['readwrite']):
            if x <= 0:
                x[...] = alpha
            else:
                x[...] = 1

    return X
