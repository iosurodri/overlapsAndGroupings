import pandas as pd
import numpy as np
import argparse
import os
import sys
import re

PATH_RESULTS = os.path.join('..', '..', 'reports', 'results')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment_folder", nargs=1, type=str, help='Name of the folder which contains the given tests')
    CLI.add_argument("--metric", nargs='*', default="all", choices=['accuracy', 'accuracy_top5', 'precision', 'recall', 'f1_score', 'all'])
    return CLI.parse_args()


def summarize_experiments(parent_folder, metric='all'):

    df_metrics = {}

    for metric in metrics:

        results_tests = {}

        parent_root = os.path.join(PATH_RESULTS, parent_folder)
        try:
            # experiments_folders = sorted(os.listdir(parent_root))
            experiments_folders = filter(lambda x: '.xlsx' not in x, sorted(os.listdir(parent_root)))
            for experiment_folder in experiments_folders:
                experiment_root = os.path.join(parent_root, experiment_folder)
                test_folders = sorted(os.listdir(experiment_root))
                results_tests[experiment_folder] = {}
                for test_folder in test_folders:
                    test_root = os.path.join(experiment_root, test_folder)
                    test_file_root = os.path.join(test_root, 'test_metrics.txt')
                    with open(test_file_root, 'r') as test_file:
                        # For 'accuracy' we just read a single value:
                        if metric == 'accuracy':
                            first_line = test_file.readline()
                            metric_val = float(re.findall("\d+\.\d+", first_line)[0])
                            results_tests[experiment_folder][test_folder] = metric_val
                        # The remaining metrics are given in format one-vs-all (as many values as classes)
                        elif metric == 'accuracy_top5':
                            line = test_file.readline()
                            # Find start of metric description:
                            while not line.startswith(metric):
                                line = test_file.readline()
                            # Read the lind of interest:
                            result_strings = re.findall("\d+\.\d+", line)
                            metric_val = float(result_strings[0])
                            results_tests[experiment_folder][test_folder] = metric_val
                        else:
                            metric_by_class = []
                            line = test_file.readline()
                            # Find start of metric description:
                            while not line.startswith(metric):
                                line = test_file.readline()
                            # Iterate over lines containing results for the given metric:
                            while ']' not in line:
                                result_strings = re.findall("\d+\.\d+", line)
                                for result_string in result_strings:
                                    metric_by_class.append(float(result_string))
                                line = test_file.readline()
                            # Read the last line:
                            result_strings = re.findall("\d+\.\d+", line)
                            for result_string in result_strings:
                                metric_by_class.append(float(result_string))

                            # Compute mean metric (using macro-averages for the metrics):
                            metric_val = np.mean(metric_by_class)
                            results_tests[experiment_folder][test_folder] = metric_val
        except FileNotFoundError as e:
            raise e
        df_results = pd.DataFrame(results_tests)
        df_results = df_results.transpose()
        df_results['mean_{}'.format(metric)] = df_results.mean(axis=1)
        df_results['std_{}'.format(metric)] = df_results.std(axis=1)
        df_metrics[metric] = df_results
    
    with pd.ExcelWriter(os.path.join(parent_root, 'metric_summary.xlsx')) as writer:    
        for metric in df_metrics.keys():
            df_metrics[metric].to_excel(writer, sheet_name=metric)


if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    experiment_folder = args.experiment_folder
    metrics = args.metric
    if metrics[0] == 'all':
        metrics = ['accuracy', 'accuracy_top5', 'precision', 'recall', 'f1_score']
    print('Starting data dumping process')
    summarize_experiments(experiment_folder[0], metrics)
    print('Excel file successfully written')