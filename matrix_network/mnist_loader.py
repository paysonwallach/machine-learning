import cPickle
import gzip
import numpy as np


def load_data():
    data = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(data)
    data.close()

    return (training_data, validation_data, test_data)


def process_data():
    training_data, validation_data, test_data = load_data()

    training_inputs = training_data[0]
    training_labels = training_data[1]
    validation_inputs = validation_data[0]
    validation_labels = validation_data[1]
    test_inputs = test_data[0]
    test_labels = test_data[1]

    for i in np.nditer(training_labels):
        training_labels[i] = vectorize_label(training_labels[i])

    training_data = zip(training_inputs, training_labels)
    validation_data = zip(validation_inputs, validation_labels)
    test_data = zip(test_inputs, test_labels)

    return (training_data, validation_data, test_data)


def vectorize_label(i):
    label = np.zeros((10, 1))
    label[i] = 1.0
    return label
