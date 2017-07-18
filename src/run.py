#! /usr/local/bin/python

import mnist_loader
import matrix_network

training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()

net = matrix_network.Network([784, 100, 30, 10])
net.evaluate(training_data, test_data, 30, 10, 3.0)
