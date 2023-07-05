# plots.py

from itertools import combinations
import os

import matplotlib.pyplot as plt

from util import *


# Draw and save the 10 feature plots
def pre_plots(data):
    features = list(data.columns)
    features.remove('species')

    if not os.path.exists(PRE_PLOTS_PATH):
        os.makedirs(PRE_PLOTS_PATH)

    for f1, f2 in combinations(features, 2):
        plt.xlabel(f1)
        plt.ylabel(f2)

        for name, group in data.groupby('species'):
            plt.scatter(group[f1], group[f2], label=name)

        plt.legend()

        path = PRE_PLOTS_PATH / f"{f1}-{f2}.png"
        plt.savefig(path, dpi=150)
        plt.clf()


def plot_with_line(model, fs, cs, save_path=None):
    f1, f2 = fs

    plt.xlabel(f1)
    plt.ylabel(f2)

    # Run the model and retreive its data
    x_test, y_test = model.x_test, model.y_test
    x0 = model.x0
    w0, w1, w2 = model.weights
    accuracy = model.accuracy

    # Calculate the y values of the line and plot it
    line_eq = lambda x: (-(w1 / w2) * x) - ((x0 * w0) / w2)
    line_ys = list(map(line_eq, x_test[f1].values))
    plt.plot(x_test[f1], line_ys)

    # Add the label column to the feature columns
    testing_data = x_test.assign(species=y_test)

    # Plot the features of each species 
    for name, group in testing_data.groupby('species'):
        plt.scatter(group[f1], group[f2], label=name)

    plt.suptitle(f"Accuracy: {accuracy}")

    plt.legend()

    # Save the plot to disk or display it
    if save_path is not None:
        path = save_path / f"{f1}-{f2} for {cs[0]}-{cs[1]}.png"
        plt.savefig(path, dpi=150)
    else:
        plt.show()

    plt.clf()


