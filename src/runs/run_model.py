# PATH workings:
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset
from src.data.save_results import log_eval_metrics
# Trainable models:
from src.models.LeNetPlus import LeNetPlus
from src.models.SupervisedNiNPlus import SupervisedNiNPlus

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

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("model_type", nargs=1, type=str, help='Type of network to run. Options are: "lenet" for '
                                                           'LeNetPlus; "nin" for SupervisedNiNPlus')
    CLI.add_argument("--dataset", nargs="?", type=str, default="CIFAR10", help='Dataset to be used for training. Options'
                                                                               'are "CIFAR10" for CIFAR10 dataset; '
                                                                               'Defaults to "CIFAR10".')
    CLI.add_argument("--name", nargs="?", type=str, help='Name for the generated files. If none, a name based on the '
                                                         'current date and time will be used instead')
    CLI.add_argument("--pool_type", nargs="?", type=str, default=None, help="Functions to be used for the pooling layer.")
    CLI.add_argument("--save_checkpoints", nargs="?", type=bool, default=False, help="""Indicates whether we will save
        the best version of the model obtained during training according to val loss, as well as the final model.""")
    CLI.add_argument("--runtype", nargs=1, type=str, default='individual', help='')
    return CLI.parse_args()


def full_process(model_type, name=None, config_file_name='default_parameters.json', dataset='CIFAR10', save_checkpoints=False,
                 pool_type='max'):

    # If no name is specified for referring to the current experiment, we generate one based on the date and hour:
    if name is None:
        date = datetime.datetime.now()
        name = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '__' + str(date.hour) + '_' + str(
            date.minute)
    # If a GPU is available, we will work with it:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # info_data will contain the hyperparameters used for training our model:
    info_data = {}

    with open(os.path.join(CONFIG_PATH, config_file_name)) as config_file:
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
        model_params = config_data['model_params'][model_type]

        info_data['model_type'] = model_type
        info_data['dataset'] = dataset
        if model_type == 'lenet':
            use_batch_norm = model_params['use_batch_norm']
            info_data['use_batch_norm'] = use_batch_norm

        scheduler_factor = model_params['scheduler_factor']
        scheduler_min_lr = model_params['scheduler_min_lr']
        optimizer = model_params['optimizer']
        learning_rate = model_params['learning_rate']
        weight_decay = model_params['weight_decay']
        momentum = model_params['momentum']

        info_data['pool_type'] = pool_type

    # 1. Data loading:
    if dataset == 'CIFAR10':
        train_dataloader, val_dataloader = load_dataset(dataset, batch_size, train=True,
                                                        train_proportion=train_proportion,
                                                        val=True, num_workers=num_workers)
        test_dataloader = load_dataset(dataset, batch_size, train=False, num_workers=num_workers)

    # 2. Model initialization:
    pool_layer = pickPoolLayer(pool_type)

    if model_type == 'lenet': 
        model = LeNetPlus(input_size, num_classes, pool_layer=pool_layer, use_batch_norm=use_batch_norm)
    elif model_type == 'nin':
        model = SupervisedNiNPlus(pool_layer, in_channels=input_size[-1], num_classes=num_classes, input_size=input_size[:-1])
    else:
        raise Exception('Non implemented yet.')
    model.to(device)

    # 3. Optimization method:
    # Optimizer initialization (SGD: Stochastic Gradient Descent):
    trainable_parameters = model.parameters()

    if optimizer == 'sgd':
        # We pass only the non frozen Parameters to the optimizer:
        optimizer = optim.SGD(trainable_parameters, lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
        # optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(trainable_parameters,
                               lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception(
            'Compatibility with the given optimizer has not been implemented yet')

    # Scheduler: On plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=5, threshold=0.0001, cooldown=0,
                                                     min_lr=scheduler_min_lr)

    # Set the loss function:
    if model_type == 'nin':
        criterion = SupervisedCrossEntropyLoss(num_epochs)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model, train_loss, train_acc, val_loss, val_acc = train(name, model, optimizer, criterion, train_dataloader, scheduler=scheduler, train_proportion=train_proportion,
                                                            batch_size=batch_size, val_loader=val_dataloader, num_epochs=num_epochs, using_tensorboard=True,
                                                            save_checkpoints=save_checkpoints)

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
    runtype = args.runtype
    model_type = args.model_type[0]
    dataset = args.dataset
    pool_type = args.pool_type
    save_checkpoints = args.save_checkpoints
    runtype = args.runtype  # ToDo: Replace for statistical tests
    full_process(model_type, name=name, dataset=dataset, pool_type=pool_type, save_checkpoints=save_checkpoints)
