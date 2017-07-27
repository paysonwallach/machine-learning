import cPickle
import gzip
import numpy as np


def load_data():
    data = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(data)
    data.close()

    training_labels = training_data[1]

    vectorized_training_labels = np.empty_like(
        training_data[1], dtype=object)

    for i in range(len(training_labels)):
        vectorized_training_labels[i] = vectorize_label(training_labels[i])

    return training_data, validation_data, test_data


def create_minibatch(data, minibatch_size):
    inputs, labels = shuffle(data)

    n = np.size(inputs[0], axis=0)

    minibatch_inputs = np.empty((n, minibatch_size))
    minibatch_labels = np.empty((minibatch_size, 1))

    print np.shape(minibatch_inputs)
    print np.shape(minibatch_labels)

    for i in range(minibatch_size):
        minibatch_inputs[:, i] = inputs[i]
        minibatch_labels[i] = labels[i]

    return minibatch_inputs, minibatch_labels


def shuffle(data):
    rng_state = np.random.get_state()
    np.random.shuffle(data[0])
    np.random.set_state(rng_state)
    np.random.shuffle(data[1])

    return data


def vectorize_label(i):
    label = np.zeros((10, 1))
    label[i] = 1.0
    return label
