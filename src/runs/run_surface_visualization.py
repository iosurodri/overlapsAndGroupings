import os
import sys

from pyrsistent import s
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.evaluate import get_prediction_metrics
from src.model_tools.load_model import load_model

from src.runs.run_test import full_test
from src.visualization.visualize_loss_surface import visualize_loss_surface

# Auxiliar modules
import torch
import argparse

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("mode", nargs=1, type=str, choices=['train', 'load'], help="""If 'train', a new model will be trained from zero. 
        If 'load', an already trained model will be loaded. In both cases, after having a loaded trained model, the visualization will start.""")
    # Parameters for mode='train':
    CLI.add_argument("--model_type", nargs="?", type=str, choices=['lenet', 'nin', 'dense', 'vgg16', 'vgg16_small'])
    CLI.add_argument("--dataset", nargs="?", type=str, choices=['CIFAR10'], default='CIFAR10')
    CLI.add_argument("--pool_type", nargs="?", type=str, default=None, help="Functions to be used for the pooling layer.")
    # Parameters for mode='load':
    
    # Parameters for both modes:
    CLI.add_argument("--name", nargs="?", type=str, help="""Name for the generated files, or the model to be loaded. If none, a name based on the 
        current date and time will be used instead""")
    
    CLI.add_argument("--config_file_name", nargs="?", type=str, default='default_parameters.json', help="config file to be used")
    return CLI.parse_args()



def run_evaluation(model_file_name, model_type='lenet', info_file_name=None, dataset='CIFAR10'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    prediction_metrics = get_prediction_metrics(model, device=device, test_loader=test_loader)
    print(prediction_metrics)


if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    mode = args.mode[0]
    if mode == 'train':
        name = args.name
        model_type = args.model_type
        dataset = args.dataset
        config_file_name = args.config_file_name
        pool_type = args.pool_type
        # Train a new model and load test data:    
        trained_model, test_loader = full_test(model_type, pool_type=pool_type, name=name, dataset=dataset, config_file_name=config_file_name, num_runs=1)
    elif mode == 'load':
        pass
    # Generate visualization of loss surface:
    visualize_loss_surface(trained_model, test_loader)