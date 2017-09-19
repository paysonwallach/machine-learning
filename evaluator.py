# Evaluates neural networks with given hyper-parameters and returns results

import numpy as np


def evaluate(network, epochs, eta, lmbd, training_data,
             validation_data=None, test_data=None):

    n_training = len(training_data)

    print "Training for {0} epochs...".format(epochs)
    for t in range(epochs):
        np.random.shuffle(training_data)

        out_str = "\tEpoch {0:2d}:".format(t+1)

        for i in range(n_training/network.minibatch_size):
            data = create_minibatch(training_data, i, network.minibatch_size,
                                    training_data=True)
            network.train(data, n_training, eta, lmbd)

        if validation_data:
            n_validation = len(validation_data)
            n_correct = 0

            for i in range(n_validation/network.minibatch_size):
                inputs, labels = create_minibatch(validation_data, i,
                                                  network.minibatch_size)
                output = network.forward_propagate(inputs)
                y_hat = np.argmax(output, axis=1)
                n_correct += np.sum(y_hat == labels)

            out_str = "{0} Training accuracy: {1:.2f}%".format(
                out_str, float(n_correct)/n_validation * 100)

        if test_data:
            n_test = len(test_data)
            n_correct = 0
            for i in range(n_test/network.minibatch_size):
                inputs, labels = create_minibatch(test_data, i,
                                                  network.minibatch_size)
                output = network.forward_propagate(inputs)
                y_hat = np.argmax(output, axis=1)
                n_correct += np.sum(y_hat == labels)

            out_str = "{0} Test accuracy: {1:.2f}%".format(
                out_str, float(n_correct)/n_test * 100)

        print out_str


def create_minibatch(data, i, minibatch_size, training_data=False):
    inputs, labels = zip(*data)

    n = np.size(inputs[0], axis=0)

    minibatch_inputs = np.zeros((minibatch_size, n))
    if training_data:
        minibatch_labels = np.empty((minibatch_size, 10))

        for j in range(minibatch_size):
            minibatch_inputs[j, :] = inputs[i+j]
            minibatch_labels[j, :] = labels[i+j]

    else:
        minibatch_labels = np.empty(minibatch_size)

        for j in range(minibatch_size):
            minibatch_inputs[j, :] = inputs[i+j]
            minibatch_labels[j] = int(labels[i+j])

    return minibatch_inputs, minibatch_labels
