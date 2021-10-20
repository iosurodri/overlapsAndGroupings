import argparse
import os
import sys
from shutil import copyfile

PATH_RUNS = os.path.join('..', '..', 'reports', 'runs')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment_folder", nargs=1, type=str, help='Name of the folder which contains the given tests')
    return CLI.parse_args()


def summarize_experiments(parent_folder):

    parent_root = os.path.join(PATH_RUNS, parent_folder)
    target_folder = os.path.join(parent_root, 'tests')
    try:
        os.mkdir(target_folder)
    except FileExistsError as e:
        pass

    try:
        # experiments_folders = sorted(os.listdir(parent_root))
        experiments_folders = filter(lambda x: x != 'tests', sorted(os.listdir(parent_root)))
        base_name = ''
        for experiment_folder in experiments_folders:
            experiment_name = base_name + experiment_folder
            experiment_root = os.path.join(parent_root, experiment_folder)
            test_folders = sorted(os.listdir(experiment_root))
            for test_folder in test_folders:
                test_name = experiment_name + '_' + test_folder
                test_root = os.path.join(experiment_root, test_folder)

                test_run_files = os.listdir(test_root)
                test_run_file_name = sorted(test_run_files, reverse=True, key=lambda x: os.path.getsize(os.path.join(test_root, x)))[0]
                test_run_file = os.path.join(test_root, test_run_file_name)

                target_test = os.path.join(target_folder, test_name)
                try:
                    os.mkdir(target_test)
                except FileExistsError as e:
                    pass

                copyfile(test_run_file, os.path.join(target_test, test_run_file_name))

    except FileNotFoundError as e:
        raise e


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
    print('Runs files copied')