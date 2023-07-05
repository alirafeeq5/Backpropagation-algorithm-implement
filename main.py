# main.py

from itertools import combinations
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from model import Model
from plots import *
from util import *
from gui import run_gui

def run_all(data):
    features = list(data.columns.values)
    species = list(data['species'].unique())

    features.remove('species')

    if not os.path.exists(POST_PLOTS_PATH):
        os.makedirs(POST_PLOTS_PATH)

    acc_sum = 0

    # Combine each combination of features with every combination of species
    for f1, f2 in combinations(features, 2):
        for s in combinations(species, 2):
            model = run(data, [f1, f2], s)
            plot_with_line(model, [f1, f2], s, POST_PLOTS_PATH)
            acc_sum += model.accuracy

    print(f"Average Accuracy: {acc_sum / 30}")


def run(data, features, species, eta=0.4, epochs=1000, bias_flag=True, mse_threshold=0.1):
    # Filter the data by species and extract the features
    filt = data['species'].isin(species)
    sel_data = data.loc[filt, list(features) + ['species']]

    # Partition the data
    x_train, y_train, x_test, y_test = partition_data(sel_data, 'species')
    logging.info(f"Extracted features: {features} for species {species}")

    model = Model(x_train, y_train, x_test, y_test, species, eta, epochs, bias_flag, mse_threshold)
    logging.info('Initialized the model')

    model.train()
    logging.info('Trained the model')

    acc = model.test()
    logging.info(f'Model accuracy = {acc}')

    print(acc)

    return model


if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    logging.info('Starting...')

    plt.style.use('ggplot')

    data = pd.read_csv('penguins.csv')
    preprocess(data)

    logging.info('Preprocessing Done.')

    # Main Code Goes Here
    run_gui(run)

    logging.info('Finished.')
