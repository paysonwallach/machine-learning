# Standard libraries
import io
import os
import cPickle

# Third-party libraries
import numpy as np
import pandas as pd


PATH = 'E:/Rogue APs/data/'


def create_labels():
    for filename in os.listdir(PATH):
        # Parse location from filename
        location = filename.split('_', 1)[0]

        # Read in weekly comparison for location
        weekly_comparison = pd.read_excel(PATH + 'Weekly Comparisons/' +
                                          filename,
                                          sheetname='Summary Data')

        # Create list of weeks in file
        weeks = weekly_comparison.DATE.unique()
        for week in weeks:
            # Create dataframe of ssids and labels in week
            labeled_ssids = weekly_comparison.loc[
                weekly_comparison.DATE == week]

            # Drop week column and write dataframe to csv
            labeled_ssids.drop('DATE', axis=1, inplace=True)

            labeled_ssids.to_csv(PATH + 'Labels/' + location +
                                 '_' + week + '.csv', sep=',',
                                 index=False)


def get_raw_data():
    raw_data_path = PATH + 'Raw Data/'
    raw_data_errors = []
    label_errors = []

    # Parse filenames for list of weeks
    weeks = [filename for filename in os.listdir(raw_data_path)
             if not filename.startswith('.')]

    for week in weeks:
        for raw_data_filename in os.listdir(raw_data_path + week):
            # Parse name of location from filename
            location = raw_data_filename.split('_')[1]

            # Create filepath to raw data
            raw_data_filepath = raw_data_path + week + '/' + raw_data_filename

            try:
                raw_data = pd.read_csv(raw_data_filepath, sep=',', header=7,
                                       encoding='latin1',
                                       error_bad_lines=False)

                raw_data = raw_data.filter(items=['Rogue MAC Address',
                                                  'Detecting AP Name',
                                                  'SSID'], axis=1)

            except FileNotFoundError:
                raw_data_errors.append("{0} does not exist".format(
                    raw_data_filepath))

                continue  # Break current loop and begin next iteration

            if location == 'North':
                # List of weekly comparison locations in raw data's
                # location
                dorms = ['Manzy', 'PVE', 'PVW', 'San Pablo', 'Towers']

            elif location == 'South':
                # List of weekly comparison locations in raw data's
                # location
                dorms = ['Aldelphi', 'Sonora']

            elif location == 'DPc':
                # List of weekly comparison locations in raw data's
                # location
                dorms = ['Downtown']

            else:
                # List of weekly comparison locations in raw data's
                # location
                dorms = [location]

            # Create dataframe of labels for all dorms in location
            label_data = pd.DataFrame()

            for dorm in dorms:
                try:
                    labels_to_append = pd.read_csv(
                        PATH + 'Labels/' + dorm + '_' + week + '.csv',
                        sep=',', header=0, names=['SSID', 'Label'],
                        encoding='latin1',
                        error_bad_lines=False)

                except FileNotFoundError:
                    label_errors.append("{0} does not exist".format(
                        raw_data_filepath))

                    continue  # Break current loop and begin next iteration

                # Append label data for dorm to dataframe of location
                label_data = label_data.append(labels_to_append,
                                               ignore_index=True)

            if label_data.empty is False:
                # Append labels to raw training data
                label_raw_data(raw_data, label_data, location, week)

            else:
                continue

    # Write error logs to csv
    raw_data_error_log = pd.DataFrame(raw_data_errors, columns=['Files'])

    label_error_log = pd.DataFrame(label_errors, columns=['Files'])

    raw_data_error_log.to_csv(PATH + "Labeled Raw Data/Raw data error log.csv")

    label_error_log.to_csv(PATH + "Labeled Raw Data/Label error log.csv")


def label_raw_data(raw_data, labels, location, week):
    # Copy dataframe without rows without ssid
    raw_data = raw_data[raw_data['SSID'] != '']

    labeled_raw_data = raw_data.merge(labels, on='SSID', sort=True, copy=False)

    # Renaming location for clarity
    if location == 'DPc':
        location = 'Downtown'

    labeled_raw_data.to_csv(PATH + 'Labeled Raw Data/' + location + '_' +
                            week + '.csv', sep=',', index=False)


def create_training_data():
    labeled_raw_data_path = PATH + 'Labeled Raw Data/'

    # Create empty lists for input and labels
    training_data = []

    # Parse filenames for list of locations
    locations = [filename for filename in os.listdir(
                 labeled_raw_data_path) if not filename.endswith('.csv')]

    for location in locations:
        location_path = labeled_raw_data_path + location + '/'
        # Parse filenames for list of weeks
        weeks = [filename.split('_')[1] for filename in
                 os.listdir(labeled_raw_data_path + location)]

        # Starting with the most recent week
        for i in range(len(weeks)-1, len(weeks) % 2, -2):

            current_week = pd.read_csv(location_path + location + '_' +
                                       weeks[i], sep=',', encoding='latin1',
                                       error_bad_lines=False)

            previous_week = pd.read_csv(location_path + location + '_' +
                                        weeks[i-1], sep=',', encoding='latin1',
                                        error_bad_lines=False)

            inputs, labels = parse_labels(
                current_week, previous_week, training_data=True)

            for i in range(len(labels)):

                training_input = inputs[:, i]
                label = labels[i, :]

                training_data.append((training_input, label))

    output = io.open(os.path.join(PATH + 'training Data/training_data.pkl'),
                     'wb')

    cPickle.dump(training_data, output)

    output.close()


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
                 "EXTERNAL (Local businesses)": 3, "UNKNOWN": 4}

    vectorized_label[translate[label.strip(' ')]] = 1

    return vectorized_label
