import numpy as np


def minibatcher(data, epochs, minibatch_size):
    for i in range(epochs):
        inputs, labels = shuffle(data)

        n = np.size(inputs)
        n_minibatches = n/minibatch_size

        minibatches = np.empty(n_minibatches)

        for i in range(n_minibatches):
            minibatches[i] = create_minibatch(data, i, minibatch_size)


def create_minibatch(data, i, minibatch_size):
    inputs, labels = shuffle(data)

    minibatch_inputs = np.empty_like(inputs)
    minibatch_labels = np.empty_like(labels)

    for j in range(minibatch_size):
        minibatch_inputs[:, j] = inputs[i+j]
        minibatch_labels[j] = labels[i+j]

    return minibatch_inputs, minibatch_labels


def shuffle(data):
    rng_state = np.random.get_state()
    np.random.shuffle(data[0])
    np.random.set_state(rng_state)
    np.random.shuffle(data[1])
