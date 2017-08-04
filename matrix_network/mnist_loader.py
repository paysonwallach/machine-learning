import cPickle
import gzip
import numpy as np


def load_data():
    data = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(data)
    data.close()

    """training_inputs, training_labels = shuffle(training_data)
    validation_data = shuffle(validation_data)
    test_data = shuffle(test_data)"""

    training_inputs, training_labels = training_data

    vectorized_labels = np.zeros((len(training_labels), 10))

    for i in range(len(training_labels)):
        vectorized_labels[i] = vectorize_label(training_labels[i])

    training_data = zip(training_inputs, vectorized_labels)
    validation_data = zip(*validation_data)
    test_data = zip(*test_data)

    return training_data, validation_data, test_data


def vectorize_label(i):
    vectorized_label = np.zeros((10))
    vectorized_label[i] = 1
    return vectorized_label
