import pandas as pd
import json
import os
from matplotlib import pyplot as plt
import seaborn as sns


def plot_predictions():
    run = 'run-2025-06-13 03-37-24-314643'
    predictions = []
    predictions_list = sorted(os.listdir(f'run_outputs\\{run}'))
    for f in predictions_list:
        if 'json' in f and 'stacked' not in f:
            with open(f'run_outputs\\{run}\\{f}', 'r') as details:
                run_details = json.load(details)
                predictions.append((f, run_details['predictions']))
    predictions_frame = pd.DataFrame()
    for prediction in predictions:
        predictions_frame[prediction[0]] = prediction[1]
    target_column = 'y_test.json'
    predictions_frame = predictions_frame[
        [target_column] + [col for col in predictions_frame.columns if col != target_column]]
    plot_series = []
    plot_index = []
    plot_color = []
    c_list = ['black', 'red', 'orange', 'green', 'blue', 'purple']
    for sample in predictions_frame.iterrows():
        for offset, point in enumerate(sample[1]):
            plot_series.append(point)
            plot_index.append(sample[0] + 0.1*offset)
            plot_color.append(c_list[offset])
    sns.scatterplot(x=plot_index, y=plot_series, hue=plot_color)
    plt.show()

if __name__ == '__main__':
    plot_predictions()