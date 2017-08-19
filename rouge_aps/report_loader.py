import numpy as np
import pandas as pd


def load_reports():
    PATH = "E:\\Rogue APs\\data\\"
    labels = ['last_seen', 'mac_address', 'location', 'D', 'E', 'F',
              'ssid', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    # Read in raw data for current and previous weeks
    current_week = pd.read_csv(PATH + "current_week.csv", sep=',',
                               header=7, names=labels, error_bad_lines=False)
    previous_week = pd.read_csv(PATH + "previous_week.csv", sep=',',
                                header=7, names=labels, error_bad_lines=False)

    # Delete extraneous data
    current_week.drop(current_week.columns[[0, 3, 4, 5, 7, 8, 9, 10, 11, 12,
                      13, 14, 15, 16]], axis=1, inplace=True)

    previous_week.drop(previous_week.columns[[0, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       12, 13, 14, 15, 16]], axis=1, inplace=True)

    # Add index column to current week
    current_week['current_week_index'] = range(len(current_week))

    # Initilize feature matrix (40 x n)
    feature_matrix = np.zeros((40, len(current_week.index)))

    seen_before(feature_matrix, current_week, previous_week)

    ssid = current_week['ssid']
    mac_address = current_week['mac_address']

    for row in range(len(current_week)):
        feature_matrix[:32, row] = ssid_to_dec(ssid[row])

        feature_matrix[34:, row] = mac_to_dec(mac_address[row])

    return feature_matrix


def seen_before(feature_matrix, current_week, previous_week):
    seen_before = current_week.merge(previous_week, on=['mac_address'])

    current_week_location = seen_before['location_x']
    previous_week_location = seen_before['location_y']

    feature_matrix_index = seen_before['current_week_index']

    for row in range(len(seen_before)):
        feature_matrix[32, feature_matrix_index[row]] = 1
        if current_week_location[row] == previous_week_location[row]:
            feature_matrix[33, feature_matrix_index[row]] = 1


def ssid_to_dec(ssid):
    ssid = str(ssid)
    dec_ssid = np.empty(32)
    for i in range(len(ssid)):
        dec_ssid[i] = ord(ssid[i]) / 128.0  # Normalize value

    return dec_ssid


def mac_to_dec(mac_address):
    remove_colons = str.maketrans(dict.fromkeys(':'))
    dec_mac = np.empty(6)
    str_mac = mac_address.translate(remove_colons)
    for i in range(6):
        dec_mac[i] = int(str_mac[i].translate(remove_colons), 16) / \
            16.0  # Normalize value

    return dec_mac
