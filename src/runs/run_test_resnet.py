# PATH workings:
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset
from src.data.save_results import log_eval_metrics
# Trainable models:
from src.models.RESNetPlus import get_resnet

# Model interaction:
from src.model_tools.train import train
from src.model_tools.evaluate import get_prediction_metrics
from src.model_tools.save_model import save_model
from src.model_tools.load_model import load_model

from src.layers.pooling_layers import pickPoolLayer

from src.functions.loss_functions import SupervisedCrossEntropyLoss

import json
import argparse
import datetime
import torch
import torch.optim as optim

# Define paths for parent directories:
CONFIG_PATH = os.path.join('..', '..', 'config')
MODELS_PATH = os.path.join('..', '..', 'reports', 'models')
RESULTS_PATH = os.path.join('..', '..', 'reports', 'results')
RUNS_PATH = os.path.join('..', '..', 'reports', 'runs')

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("model_type", nargs=1, type=str, choices=['small', 'pool'], help='Type of RESNet model to run.')
    CLI.add_argument("--bottleneck", nargs="?", type=str, choices=['pad_constant', 'pad_replicate', 'conv_bottleneck'])
    CLI.add_argument("--planes", nargs="?", type=str, choices=['double', 'constant_16', 'constant_32'])
    CLI.add_argument("--model_size", nargs="?", type=int, choices=[20, 32, 56], default=20)

    CLI.add_argument("--dataset", nargs="?", type=str, default="CIFAR10", help='Dataset to be used for training. Options'
                                                                               'are "CIFAR10" for CIFAR10 dataset; '
                                                                               'Defaults to "CIFAR10".')
    CLI.add_argument("--name", nargs="?", type=str, help='Name for the generated files. If none, a name based on the '
                                                         'current date and time will be used instead')
    CLI.add_argument("--pool_type", nargs="?", type=str, default=None, help="Functions to be used for the pooling layer.")
    CLI.add_argument("--num_runs", nargs="?", type=int, default=5, help="Number of tests to be performed. Defaults to 5.")
    CLI.add_argument("--initial_pool_exp", nargs="?", type=float, default=None, help='''If pool_type requires it, sets the 
        initial value for the weight (exponent) ''')
    CLI.add_argument("--save_checkpoints", nargs="?", type=bool, default=False, help="""Indicates whether we will save
        the best version of the model obtained during training according to val loss, as well as the final model.""")
    CLI.add_argument("--log_param_dist", nargs="?", type=bool, default=False, help="""Indicates whether the distribution
        of custom learnable parameters are logged (using tensorboard) or not.""")
    CLI.add_argument("--log_grad_dist", nargs="?", type=bool, default=False, help="""Indicates whether the distribution of 
        gradients for convolutional layers are logged (using tensorboard) or not.""")
    CLI.add_argument("--config_file_name", nargs="?", type=str, default='default_parameters.json', help="config file to be used")
    CLI.add_argument("--learning_rate", nargs="?", type=float, default=None)
    return CLI.parse_args()


def full_test(model_type, name=None, config_file_name='default_parameters.json', dataset='CIFAR10', save_checkpoints=False, log_param_dist=False, 
    log_grad_dist=False, pool_type=None, bottleneck='pad_constant', planes='double', model_size=20, num_runs=5, initial_pool_exp=None,
    learning_rate=None):

    # If no name is specified for referring to the current experiment, we generate one based on the date and hour:
    if name is None:
        date = datetime.datetime.now()
        name = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '__' + str(date.hour) + '_' + str(
            date.minute)
    # If a GPU is available, we will work with it:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # info_data will contain the hyperparameters used for training our model:
    info_data = {}

    with open(os.path.join(CONFIG_PATH, 'resnet', config_file_name)) as config_file:
        config_data = json.load(config_file)
        # Train loop configuration:
        train_params = config_data['train_params']
        num_epochs = train_params['num_epochs']
        batch_size = train_params['batch_size']

        dataset_params = config_data['dataset_params'][dataset]
        input_size = dataset_params['input_size']
        info_data['input_size'] = input_size
        num_classes = dataset_params['num_classes']
        info_data['num_classes'] = num_classes
        train_proportion = dataset_params['train_proportion']
        num_workers = dataset_params['num_workers']
        model_params = config_data['model_params']

        info_data['model_type'] = model_type
        info_data['dataset'] = dataset

        scheduler_factor = model_params['scheduler_factor']
        scheduler_min_lr = model_params['scheduler_min_lr']
        optimizer_name = model_params['optimizer']
        if learning_rate is None:
            learning_rate = model_params['learning_rate']
        weight_decay = model_params['weight_decay']
        momentum = model_params['momentum']

        # Hyperparameters of the test:
        info_data['pool_type'] = pool_type
        info_data['bottleneck'] = bottleneck
        info_data['planes'] = planes
        info_data['model_size'] = model_size


    # Create folders for reports associated to test if not existant:
    try: 
        os.mkdir(os.path.join(RESULTS_PATH, name))
    except:
        pass
    try:
        os.mkdir(os.path.join(RUNS_PATH, name))
    except:
        pass
    try:
        os.mkdir(os.path.join(MODELS_PATH, name))
    except:
        pass

    original_name = name

    for test_idx in range(num_runs):
        name = os.path.join(original_name, 'test_{}'.format(test_idx))
        # 1. Data loading:
        if dataset == 'CIFAR10':
            train_dataloader, val_dataloader = load_dataset(dataset, batch_size, train=True,
                                                            train_proportion=train_proportion,
                                                            val=True, num_workers=num_workers)
            test_dataloader = load_dataset(dataset, batch_size, train=False, num_workers=num_workers)

        # 2. Model initialization:
        pool_layer = None
        if pool_type is not None:
            pool_layer = pickPoolLayer(pool_type)

        # TODO: Replace models available by RESNet types:

        model = get_resnet(model_type, pool_layer=pool_layer, bottleneck_option=bottleneck, planes_conf=planes, size=model_size)
        model.to(device)

        # 3. Optimization method:
        # Optimizer initialization (SGD: Stochastic Gradient Descent):
        trainable_parameters = model.parameters()

        if optimizer_name == 'sgd':
            # We pass only the non frozen Parameters to the optimizer:
            optimizer = optim.SGD(trainable_parameters, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
            # optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.Adam(trainable_parameters,
                                lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception(
                'Compatibility with the given optimizer has not been implemented yet')

        # Scheduler: On plateau
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=5, threshold=0.0001, cooldown=0,
        #                                                 min_lr=scheduler_min_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Set the loss function:
        criterion = torch.nn.CrossEntropyLoss()

        model, train_loss, train_acc, val_loss, val_acc = train(name, model, optimizer, criterion, train_dataloader, scheduler=scheduler, train_proportion=train_proportion,
                                                                batch_size=batch_size, val_loader=val_dataloader, num_epochs=num_epochs, using_tensorboard=True,
                                                                save_checkpoints=save_checkpoints, log_param_dist=log_param_dist, log_grad_dist=log_grad_dist)

        # log_eval_results(name, val_acc, loss=val_loss)
        metrics = get_prediction_metrics(model, device, test_dataloader, verbose=False)
        log_eval_metrics(name, metrics)
        save_model(model, name, info_data)
        if save_checkpoints:
            model = load_model(os.path.join(MODELS_PATH, name + '_checkpoint'), model_type=model_type, info_data=info_data).to(device)
            metrics_best = get_prediction_metrics(model, device=device, test_loader=test_dataloader, verbose=False)
            log_eval_metrics(name + '_best', metrics_best)


if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    name = args.name
    model_type = args.model_type[0]
    bottleneck = args.bottleneck
    planes = args.planes
    model_size = args.model_size

    dataset = args.dataset
    num_runs = args.num_runs
    config_file_name = args.config_file_name
    pool_type = args.pool_type
    initial_pool_exp = args.initial_pool_exp
    save_checkpoints = args.save_checkpoints
    log_param_dist = args.log_param_dist
    log_grad_dist = args.log_grad_dist

    learning_rate = args.learning_rate
    full_test(model_type, name=name, dataset=dataset, bottleneck=bottleneck, planes=planes, model_size=model_size, 
        pool_type=pool_type, num_runs=num_runs, save_checkpoints=save_checkpoints, config_file_name=config_file_name, 
        log_param_dist=log_param_dist, log_grad_dist=log_grad_dist, initial_pool_exp=initial_pool_exp,
        learning_rate=learning_rate)
