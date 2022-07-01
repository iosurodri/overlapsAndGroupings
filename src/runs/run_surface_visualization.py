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

    # Configuration parameters:
    CLI.add_argument("--plot_graphs", nargs="?", type=bool, default=False, help="If false, only csv containing the values of the loss surfaces will be saved.")
    CLI.add_argument("--graph_3d", nargs="?", type=bool, default=False, help="If true, 3d surface plots will be graphed (as well as contour plots).")

    # Parameters for mode='train':
    CLI.add_argument("--pool_type", nargs="?", type=str, default=None, help="Functions to be used for the pooling layer.")
    CLI.add_argument("--config_file_name", nargs="?", type=str, default='default_parameters.json', help="config file to be used")

    # Parameters for mode='load':
    CLI.add_argument("--batch_size", nargs="?", type=int, default=64)
    CLI.add_argument("--num_workers", nargs="?", type=int, default=2)
    CLI.add_argument("--test_idx", nargs="?", type=int, default=0)
    
    # Parameters for both modes:
    CLI.add_argument("--model_type", nargs="?", type=str, choices=['lenet', 'nin', 'dense', 'vgg16', 'vgg16_small'])
    CLI.add_argument("--dataset", nargs="?", type=str, choices=['CIFAR10'], default='CIFAR10')
    CLI.add_argument("--name", nargs="?", type=str, help="""Name for the generated files, or the model to be loaded. If none, a name based on the 
        current date and time will be used instead""")
    
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
    plot_graphs = args.plot_graphs
    graph_3d = args.graph_3d
    if mode == 'train':
        name = args.name
        model_type = args.model_type
        dataset = args.dataset
        config_file_name = args.config_file_name
        pool_type = args.pool_type
        # Train a new model and load test data:    
        trained_model, test_loader = full_test(model_type, pool_type=pool_type, name=name, dataset=dataset, config_file_name=config_file_name, num_runs=1)
        test_idx = 0
    elif mode == 'load':
        name = args.name  # Must indicate the name of a model saved in /reports/models/ (not inside additional subfolders)
        test_idx = args.test_idx
        model_type = args.model_type
        dataset = args.dataset
        config_file_name = args.config_file_name
        batch_size = args.batch_size
        num_workers = args.num_workers
        # Get test dataloader:
        _, test_loader = load_dataset(dataset, batch_size=batch_size, val=False, num_workers=num_workers)
        # Load pretrained model:
        model_file_name = os.path.join(name, 'test_{}'.format(str(test_idx)))
        info_file_name = os.path.join(name, 'test_{}_info.json'.format(str(test_idx)))
        trained_model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate visualization of loss surface:
    visualize_loss_surface(trained_model, test_loader, name=name, test_idx=test_idx, plot_graphs=plot_graphs, graph_3d=graph_3d)