# Module of cost functions for use in matrix-based neural networks

import numpy as np


def quadratic(y_hat, label):
    return y_hat - label


def cross_entropy(y_hat, label):
    return np.nan_to_num(-label * np.log(y_hat) - (1 - label) *
                         np.log(1 - y_hat))
