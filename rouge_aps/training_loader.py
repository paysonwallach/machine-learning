# Third-party libraries
import numpy as np


def parse_labels(current_week, previous_week, training_data=False):
    # Extract label series
    labels = current_week.Label
    if training_data:
        labels = vectorize_labels(labels)

    current_week = current_week.filter(items=['Rogue MAC Address',
                                              'Detecting AP Name',
                                              'SSID'], axis=1)

    previous_week = previous_week.filter(items=['Rogue MAC Address',
                                                'Detecting AP Name',
                                                'SSID'], axis=1)

    # Add index column to current week
    current_week['current_week_index'] = range(len(current_week))

    feature_matrix = create_feature_matrix(current_week, previous_week)

    return feature_matrix, labels


def create_feature_matrix(current_week, previous_week):
    # Initilize feature matrix (40 x n)
    feature_matrix = np.zeros((40, len(current_week.index)))

    seen_previous_week(feature_matrix, current_week, previous_week)

    ssid = current_week['SSID']
    mac_address = current_week['Rogue MAC Address']

    for i in range(len(current_week)):
        feature_matrix[:32, i] = ssid_to_dec(ssid[i])
        feature_matrix[34:, i] = mac_to_dec(mac_address[i])

    return feature_matrix


def seen_previous_week(feature_matrix, current_week, previous_week):
    seen_previous_week = current_week.merge(previous_week,
                                            on='Rogue MAC Address',
                                            suffixes=('_current', '_previous'))

    current_week_location = seen_previous_week['Detecting AP Name_current']
    previous_week_location = seen_previous_week['Detecting AP Name_previous']

    feature_matrix_index = seen_previous_week['current_week_index']

    for i in range(len(seen_previous_week)):
        feature_matrix[32, feature_matrix_index[i]] = 1
        if current_week_location[i] == previous_week_location[i]:
            feature_matrix[33, feature_matrix_index[i]] = 1


def ssid_to_dec(ssid):
    ssid = str(ssid)
    dec_ssid = np.empty(32)
    for i in range(len(ssid)):
        dec_ssid[i] = ord(ssid[i]) / 128.0  # Normalize feature scale

    return dec_ssid


def mac_to_dec(mac_address):
    remove_colons = str.maketrans(dict.fromkeys(':'))
    dec_mac = np.empty(6)
    str_mac = mac_address.translate(remove_colons)
    for i in range(6):
        dec_mac[i] = int(str_mac[i].translate(remove_colons), 16) / \
            16.0  # Normalize feature scale

    return dec_mac


def vectorize_labels(labels):

    # Create vectorized training labels
    vectorized_labels = np.zeros((len(labels), 5))

    for i in range(len(labels)):
        vectorized_labels[i] = vectorize_label(labels[i])

    return vectorized_labels


def vectorize_label(label):
    vectorized_label = np.zeros((5))

    translate = {"ROUTER": 0, "PRINTER": 1,
                 "OTHER (Smart TV, Chromecast, Roku, etc.)": 2,
                 "EXTERNAL (Local businesses)": 3, "UNKNOWN ": 4}

    vectorized_label[translate[label]] = 1

    return vectorized_label
