import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.evaluate import get_prediction_metrics, get_probability_matrix
from src.model_tools.load_model import load_model

# Auxiliar modules
import torch

PATH_RESULTS = os.path.join('..', '..', 'reports', 'results')



def run_evaluation(model_file_name, model_type='lenet', info_file_name=None, dataset='CIFAR10'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    # test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    _, test_loader = load_dataset(dataset, batch_size=batch_size, val=False, num_workers=0)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    prediction_metrics = get_prediction_metrics(model, device=device, test_loader=test_loader)
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
    # base_folder = 'dense'
    base_folder = 'interesting_models'
    # base_model_file_name = 'dense_pool_avg'
    base_model_file_name = 'dense_pool_max'

    model_type = 'dense'
    dataset = 'CIFAR10'
    mode = 'evaluation'    
    for test_idx in range(5):
        model_file_name = os.path.join(base_folder, base_model_file_name)
        model_file_name = os.path.join(model_file_name, 'test_{}'.format(str(test_idx)))
        info_file_name = model_file_name + '_info.json'
        if mode == 'evaluation':
            run_evaluation(model_file_name, model_type=model_type, info_file_name=info_file_name, dataset=dataset)
        elif mode == 'get_prob_matrix':
            run_get_prob_matrix(base_model_file_name, model_file_name, model_type=model_type, info_file_name=info_file_name, dataset=dataset, test_idx=test_idx)