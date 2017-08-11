# Module of activation functions for use in matrix-based neural networks

import numpy as np


def sigmoid(s, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + np.exp(-s))
    else:
        return sigmoid(s) * (1.0 - sigmoid(s))


def softmax(s):
    z = np.sum(np.exp(s), axis=1)
    z = z.reshape(z.shape[0], 1)
    return np.exp(s) / z


def relu(s, deriv=False):
    if not deriv:
        for i in range(np.ma.size(s, axis=0)):
            for j in range(np.ma.size(s, axis=1)):
                if s[i, j] <= 0:
                    s[i, j] = 0

    else:
        for i in range(np.ma.size(s, axis=0)):
            for j in range(np.ma.size(s, axis=1)):
                if s[i, j] <= 0:
                    s[i, j] = 0
                else:
                    s[i, j] = 1

    return s


def prelu(s, alpha=0.1, deriv=False):
    if not deriv:
        for i in range(np.ma.size(s, axis=0)):
            for j in range(np.ma.size(s, axis=1)):
                if s[i, j] <= 0:
                    s[i, j] *= alpha

    else:
        for i in range(np.ma.size(s, axis=0)):
            for j in range(np.ma.size(s, axis=1)):
                if s[i, j] <= 0:
                    s[i, j] = alpha
                else:
                    s[i, j] = 1

    return s
