import cPickle
import gzip
import numpy as np


def load_data():
    data = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(data)
    data.close()

    training_inputs, training_labels = shuffle(training_data)
    validation_data = shuffle(validation_data)
    test_data = shuffle(test_data)

    vectorized_labels = np.zeros((len(training_labels), 10))

    for i in range(len(training_labels)):
        vectorized_labels[i] = vectorize_label(training_labels[i])

    training_data = zip(training_inputs, vectorized_labels)

    return training_data, validation_data, test_data


def shuffle(data):
    rng_state = np.random.get_state()
    np.random.shuffle(data[0])
    np.random.set_state(rng_state)
    np.random.shuffle(data[1])

    return data


def vectorize_label(i):
    vectorized_label = np.zeros((10))
    vectorized_label[i] = 1
    return vectorized_label
