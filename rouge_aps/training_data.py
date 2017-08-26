# Standard libraries
import os

# Third-party libraries
import pandas as pd

# Modules
import training_loader


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
                raw_data = pd.read_csv(raw_data_filepath, sep=',',
                                       header=7, error_bad_lines=False)

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
                        error_bad_lines=False)

                except (FileNotFoundError, UnicodeDecodeError) as error:
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

    raw_data_error_log.to_csv(PATH + "Training Data/Raw data error log.csv")

    label_error_log.to_csv(PATH + "Training Data/Label error log.csv")


def label_raw_data(raw_data, labels, location, week):
    # Copy dataframe without rows without ssid
    raw_data = raw_data[raw_data['SSID'] != '']

    labeled_raw_data = raw_data.merge(labels, on='SSID', sort=True, copy=False)

    labeled_raw_data.to_csv(PATH + 'Labeled Raw Data/' + location + '_' +
                            week + '.csv', sep=',', index=False)


def create_training_data():
    labeled_raw_data_path = PATH + 'Labeled Raw Data/'

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
            print("Current week: " + location + ' ' + weeks[i])
            current_week = pd.read_csv(location_path + location + '_' +
                                       weeks[i], sep=',', encoding='latin1',
                                       error_bad_lines=False)
            print("Previous week: " + location + ' ' + weeks[i-1])
            previous_week = pd.read_csv(location_path + location + '_' +
                                        weeks[i-1], sep=',', encoding='latin1',
                                        error_bad_lines=False)

            inputs, labels = training_loader.parse_labels(current_week,
                                                          previous_week,
                                                          training_data=True)
            print(inputs)
