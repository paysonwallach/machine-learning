# Third-party libraries
import numpy as np
import pandas as pd


def load_reports():
    PATH = "E:/Rogue APs/data/"
    labels = ['last_seen', 'mac_address', 'location', 'D', 'E', 'F',
              'ssid', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    # Read in raw data for current and previous weeks
    current_week = pd.read_csv(PATH + "current_week.csv", sep=',',
                               header=7, names=labels, error_bad_lines=False)
    previous_week = pd.read_csv(PATH + "previous_week.csv", sep=',',
                                header=7, names=labels, error_bad_lines=False)

    # Filter out extraneous columns
    current_week = current_week.filter(items=['mac_address', 'location',
                                              'ssid'], axis=1)

    previous_week = previous_week.filter(items=['mac_address', 'location',
                                                'ssid'], axis=1)

    # Add index column to current week
    current_week['current_week_index'] = range(len(current_week))

    return current_week, previous_week


def create_feature_matrix(current_week, previous_week):
    # Initilize feature matrix (40 x n)
    feature_matrix = np.zeros((40, len(current_week.index)))

    seen_previous_week(feature_matrix, current_week, previous_week)

    ssid = current_week['ssid']
    mac_address = current_week['mac_address']

    for i in range(len(current_week)):
        feature_matrix[:32, i] = ssid_to_dec(ssid[i])
        feature_matrix[34:, i] = mac_to_dec(mac_address[i])

    return feature_matrix


def seen_previous_week(feature_matrix, current_week, previous_week):
    seen_previous_week = current_week.merge(previous_week, on='mac_address',
                                            suffixes=('_current', '_previous'))

    current_week_location = seen_previous_week['location_current']
    previous_week_location = seen_previous_week['location_previous']

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
