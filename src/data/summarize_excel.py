import pandas as pd
import argparse
import os
import sys
import re

PATH_RESULTS = os.path.join('..', '..', 'reports', 'results')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment_folder", nargs=1, type=str, help='Name of the folder which contains the given tests')
    return CLI.parse_args()


def summarize_experiments(parent_folder):

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
                    first_line = test_file.readline()
                    result_test = float(re.findall("\d+\.\d+", first_line)[0])
                    results_tests[experiment_folder][test_folder] = result_test
            results_tests
    except FileNotFoundError as e:
        raise e
    df_results = pd.DataFrame(results_tests)
    df_results = df_results.transpose()
    df_results['mean_acc'] = df_results.mean(axis=1)
    df_results['std_acc'] = df_results.std(axis=1)
    with pd.ExcelWriter(os.path.join(parent_root, 'acc_summary.xlsx')) as writer:
        df_results.to_excel(writer)



if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    experiment_folder = args.experiment_folder
    print('Starting data dumping process')
    summarize_experiments(experiment_folder[0])
    print('Excel file successfully written')