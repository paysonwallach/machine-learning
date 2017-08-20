import glob
import os
import pandas as pd
import numpy as np
import report_loader

training_inputs = report_loader.load_reports()


def load_training_labels():
    PATH = "E:\\Rogue APs\\data\\"

    # Read in labeled data from past weeks
    training_labels = pd.read_csv(PATH + 'test.csv', sep=',',
                                  encoding='latin1', error_bad_lines=False)

    training_labels.sort_values(by='label', inplace=True)

    # Create vectorized training labels
    labels = training_labels['label']

    vectorized_labels = np.zeros((len(labels), 5))

    for i in range(len(labels)):
        vectorized_labels[i] = vectorize_label(labels[i])

    print(vectorized_labels)

    return training_labels


def vectorize_label(label):
    vectorized_label = np.zeros((5))
    translate = {"Router": 0, "Local Business": 1, "Other": 2, "Printer": 3,
                 "Unknown": 4}
    vectorized_label[translate[label]] = 1
    return vectorized_label
