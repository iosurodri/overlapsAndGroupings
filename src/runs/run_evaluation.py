import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import argparse

# Data loading and saving:
from src.data.load_data import load_dataset
from src.data.save_results import log_eval_metrics

# Model interaction:
from src.model_tools.evaluate import get_prediction_metrics, get_probability_matrix
from src.model_tools.load_model import load_model

# Auxiliar modules
import torch

PATH_RESULTS = os.path.join('..', '..', 'reports', 'results')

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("base_folder", nargs=1, type=str)
    CLI.add_argument("base_model_file_name", nargs=1, type=str)
    CLI.add_argument("--model_type", nargs="?", type=str, choices=['lenet', 'nin', 'resnet', 'vgg16_small'])
    CLI.add_argument("--dataset", nargs="?", type=str, default="CIFAR100")
    CLI.add_argument("--mode", nargs="?", type=str, default="evaluation", choices=['evaluation', 'get_prob_matrix'])
    CLI.add_argument("--num_tests", nargs="?", type=int, default=1)
    
    return CLI.parse_args()


def run_evaluation(model_file_name, model_type='lenet', info_file_name=None, dataset='CIFAR10', results_file_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    # test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    _, test_loader = load_dataset(dataset, batch_size=batch_size, val=False, num_workers=0)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    prediction_metrics = get_prediction_metrics(model, device=device, test_loader=test_loader)
    # Log metrics:
    if results_file_path is not None:
        log_eval_metrics(results_file_path, prediction_metrics)
    print(prediction_metrics)


def run_get_prob_matrix(base_model_file_name, model_file_name, model_type='lenet', test_idx=0, info_file_name=None, dataset='CIFAR10'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    # test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    _, test_loader = load_dataset(dataset, batch_size=batch_size, val=False, num_workers=0)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    path_file_results = os.path.join(PATH_RESULTS, base_model_file_name)
    try:
        os.mkdir(path_file_results)
    except FileExistsError:
        pass
    path_file_results = os.path.join(path_file_results, 'test_{}'.format(str(test_idx)))
    get_probability_matrix(path_file_results, model, device=device, test_loader=test_loader)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    base_folder = args.base_folder[0]
    base_model_file_name = args.base_model_file_name[0]
    # base_folder = 'dense'
    # base_folder = 'interesting_models'
    # base_model_file_name = 'dense_pool_avg'
    # base_model_file_name = 'dense_pool_max'
    model_type = args.model_type
    dataset = args.dataset
    mode = args.mode
    num_tests = args.num_tests
    # model_type = 'dense'
    # dataset = 'CIFAR10'
    # mode = 'evaluation'
    for test_idx in range(num_tests):
        model_file_name = os.path.join(base_folder, base_model_file_name)
        model_file_name = os.path.join(model_file_name, 'test_{}'.format(str(test_idx)))
        info_file_name = model_file_name + '_info.json'
        
        results_folder_path = os.path.join(PATH_RESULTS, base_model_file_name)
        try:
            os.mkdir(results_folder_path)
        except:
            pass
        if mode == 'evaluation':
            run_evaluation(model_file_name, model_type=model_type, info_file_name=info_file_name, dataset=dataset, results_file_path=os.path.join(results_folder_path, 'test_{}'.format(str(test_idx))))
        elif mode == 'get_prob_matrix':
            run_get_prob_matrix(base_model_file_name, model_file_name, model_type=model_type, info_file_name=info_file_name, dataset=dataset, test_idx=test_idx)