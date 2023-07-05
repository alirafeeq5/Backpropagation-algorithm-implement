# util.py

from pathlib import Path

import numpy as np
import pandas as pd

PRE_PLOTS_PATH = Path('./pre_plots/')
POST_PLOTS_PATH = Path('./post_plots/')


def partition_data(data, y_label):
    # Create empty DataFrames
    training = pd.DataFrame()
    testing = pd.DataFrame()

    # Group by the label, then add 30 training rows to the training DataFrame
    # and 20 test rows to the testing frame
    for _, group in data.groupby(y_label):
        training = pd.concat([training, group.iloc[:29]], ignore_index=True)
        testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

    # Randomly shuffle the data
    training = training.sample(frac=1)
    testing = testing.sample(frac=1)

    # Return the feature columns and label column separately for each DataFrame, such that the return values
    # are training_feature_columns, training_label_column, testing_feature_columns, testing_label_coloumn
    # Inspired by scikit-learn train_test_split function
    return (
        training.loc[:, training.columns != y_label],
        training[y_label],
        testing.loc[:, testing.columns != y_label],
        testing[y_label]
    )


def preprocess(data):
    # Replace Na values with the last valid value of their column
    data.fillna('pad', inplace=True)
    # Convert `male` to 1, `female` to 0
    data['gender'] = np.where(data['gender'] == 'male', 1, 0)
    # Convert `flipper_length_mm` to Cm 
    data['flipper_length_mm'] = data['flipper_length_mm'] / 10
    # Convert `body_mass_g` to Kg
    data['body_mass_g'] = data['body_mass_g'] / 1000

    # Convert all values to fractions
    func = lambda x: x / 100
    data['flipper_length_mm'] = data['flipper_length_mm'].apply(func)
    data['bill_length_mm'] = data['bill_length_mm'].apply(func)
    data['bill_depth_mm'] = data['bill_depth_mm'].apply(func)
    data['body_mass_g'] = data['body_mass_g'].apply(func)

