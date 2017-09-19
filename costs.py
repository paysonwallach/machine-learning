# Module of cost functions for use in matrix-based neural networks

import numpy as np


class quadratic(object):
    @staticmethod
    def func(y_hat, label):
        # Return cost of actual output "y_hat" with respect to expected
        # output "label"
        return 0.5*np.linalg.norm(y_hat - label)**2

    @staticmethod
    def error(y_hat, label):
        # Return the error signal "delta" from the output layer
        return (y_hat - label)


class cross_entropy(object):
    @staticmethod
    def func(y_hat, label):
        # Return cost of actual output "y_hat" with respect to expected
        # output "label"
        return np.sum(np.nan_to_num(-label * np.log(y_hat) - (1 - label) *
                                    np.log(1 - y_hat)))

    @staticmethod
    def error(y_hat, label):
        # Return the error signal "delta" from the output layer
        return (y_hat - label)
