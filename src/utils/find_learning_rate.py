# PATH workings:
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset

# Trainable models:
from src.models.LeNetPlus import LeNetPlus
from src.models.SupervisedNiNPlus import SupervisedNiNPlus
from src.models.DenseNetPlus import DenseNetPlus
from src.models.VGG import vgg16_bn, vgg16_bn_small

# Model interaction:
from src.model_tools.train import train

from src.layers.pooling_layers import GroupingPlusPool2d, GroupingCombPool2d, pickPoolLayer

from src.functions.loss_functions import SupervisedCrossEntropyLoss

from src.optim.optim import get_param_groups

import json
import argparse
import datetime
import torch
import torch.optim as optim

from tqdm import tqdm

# Define paths for parent directories:
CONFIG_PATH = os.path.join('..', '..', 'config')
RUNS_PATH = os.path.join('..', '..', 'reports', 'runs')

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("model_type", nargs=1, type=str, help='Type of network to run. Options are: "lenet" for '
                                                           'LeNetPlus; "nin" for SupervisedNiNPlus')
    CLI.add_argument("--name", nargs="?", type=str, help='Name for the generated files. If none, a name based on the '
                                                         'current date and time will be used instead')
    CLI.add_argument("--dataset", nargs="?", type=str, default="CIFAR10", help='Dataset to be used for training. Options'
                                                                               'are "CIFAR10" for CIFAR10 dataset; '
                                                                               'Defaults to "CIFAR10".')
    CLI.add_argument("--pool_type", nargs="?", type=str, default=None, help="Functions to be used for the pooling layer.")
    CLI.add_argument("--config_file_name", nargs="?", type=str, default='default_parameters.json', help="config file to be used")
    
    CLI.add_argument("--initial_lr", nargs="?", type=float, default=0.1, help="""Initial value for learning rate.""")
    CLI.add_argument("--reduction_factor", nargs="?", type=float, default=0.5, help="""Each following learning rate candidate
        will be multiplied by reduction_factor.""")
    CLI.add_argument("--num_tries", nargs="?", type=int, default=5, help="""Number of learning rates to test.""")
    CLI.add_argument("--threshold", nargs="?", type=float, help="""Validation accuracy will be compared
        with this threshold. If the accuracy is bigger than the threshold, the current learning_rate will be reported""")
    
    CLI.add_argument("--log_param_dist", nargs="?", type=bool, default=False, help="""Indicates whether the distribution
        of custom learnable parameters are logged (using tensorboard) or not.""")
    CLI.add_argument("--log_grad_dist", nargs="?", type=bool, default=False, help="""Indicates whether the distribution of 
        gradients for convolutional layers are logged (using tensorboard) or not.""")
    CLI.add_argument("--log_first_epoch", nargs="?", type=bool, default=False, help="""Indicates whether logs will be generated for all
        batches of first iteration or not.""")
    return CLI.parse_args()


def find_best_lr(model_type, name=None, dataset='CIFAR10', config_file_name='default_parameters.json', 
    pool_type='max', threshold=0.15, initial_lr=0.1, reduction_factor=0.5, num_tries=5,
    log_param_dist=False, log_grad_dist=False, log_first_epoch=False):

    # If no name is specified for referring to the current experiment, we generate one based on the date and hour:
    if name is None:
        date = datetime.datetime.now()
        name = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '__' + str(date.hour) + '_' + str(
            date.minute)
    # If a GPU is available, we will work with it:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(CONFIG_PATH, config_file_name)) as config_file:
        config_data = json.load(config_file)
        # Train loop configuration:
        train_params = config_data['train_params']
        num_epochs = train_params['num_epochs']
        batch_size = train_params['batch_size']

        dataset_params = config_data['dataset_params'][dataset]
        input_size = dataset_params['input_size']
        num_classes = dataset_params['num_classes']
        train_proportion = dataset_params['train_proportion']
        num_workers = dataset_params['num_workers']
        model_params = config_data['model_params'][model_type]

        if model_type == 'lenet':
            use_batch_norm = model_params['use_batch_norm']

        optimizer_name = model_params['optimizer']
        weight_decay = model_params['weight_decay']
        momentum = model_params['momentum']


    # Create folders for reports associated to test if not existant:
    try:
        os.mkdir(os.path.join(RUNS_PATH, 'tuning'))
    except:
        pass
    name = os.path.join('tuning', name)
    try:
        os.mkdir(os.path.join(RUNS_PATH, name))
    except:
        pass

    original_name = name

    # for test_idx in range(num_tries):
    name = os.path.join(original_name, 'test')
    # 1. Data loading:
    # if dataset == 'CIFAR10':
    train_dataloader, val_dataloader, _ = load_dataset(dataset, batch_size, 
                                                        train_proportion=train_proportion, 
                                                        val=True, num_workers=num_workers)
    
    # 2. Model initialization:
    pool_layer = pickPoolLayer(pool_type)

    #########################################
    #### LOOP TO FIND BEST LEARNING_RATE #### 
    #########################################

    # initial_lr=0.1, reduction_factor=0.5, n_tries=5
    possible_lr = [initial_lr * reduction_factor**i for i in range(num_tries)]
    for current_lr in possible_lr:

        print('### TESTING LEARNING RATE {} ###'.format(str(current_lr)))

        if model_type == 'lenet': 
            model = LeNetPlus(input_size, num_classes, pool_layer=pool_layer, use_batch_norm=use_batch_norm)
        elif model_type == 'nin':
            model = SupervisedNiNPlus(pool_layer, in_channels=input_size[-1], num_classes=num_classes, input_size=input_size[:-1], initial_pool_exp=initial_pool_exp)
        elif model_type == 'dense':
            model = DenseNetPlus(pool_layer=pool_layer, in_channels=input_size[-1], num_classes=num_classes)
        elif model_type == 'vgg16':
            model = vgg16_bn(pool_layer=pool_layer, num_classes=num_classes)
        elif model_type == 'vgg16_small':
            model = vgg16_bn_small(pool_layer=pool_layer, num_classes=num_classes)
        else:
            raise Exception('Non implemented yet.')
        model.to(device)

        # 3. Optimization method:
        # Optimizer initialization (SGD: Stochastic Gradient Descent):
        # Get parameters (both standard and custom ones)
        if optimizer_name == 'sgd':
            # We pass only the non frozen Parameters to the optimizer:
            optimizer = optim.SGD(model.parameters(), lr=current_lr,
                                    momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=current_lr,
                                    momentum=momentum, weight_decay=weight_decay)
        else:
            raise Exception(
                'Compatibility with the given optimizer has not been implemented yet')

        # Scheduler: On plateau
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=5, threshold=0.0001, cooldown=0,
        #                                                 min_lr=scheduler_min_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Set the loss function:
        if model_type == 'nin':
            criterion = SupervisedCrossEntropyLoss(num_epochs)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model, _, _, _, val_acc = train(name, model, optimizer, criterion, train_dataloader, scheduler=scheduler, train_proportion=train_proportion,
                                        batch_size=batch_size, val_loader=val_dataloader, num_epochs=1, using_tensorboard=True,
                                        save_checkpoints=False, log_param_dist=log_param_dist, log_grad_dist=log_grad_dist,
                                        log_first_epoch=log_first_epoch)
        
        print('Finished test with learning rate {} - Val accuracy: {}'.format(str(current_lr), str(val_acc[0])))
        if val_acc[0] > threshold:
            break

    if val_acc[0] > threshold:
        print('Training correct for learning rate: {}'.format(str(current_lr)))
    else:
        print('Unable to find a good value for learning rate')


if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]

    args = parse_args()
    model_type = args.model_type[0]
    name = args.name
    dataset = args.dataset
    config_file_name = args.config_file_name
    pool_type = args.pool_type
    
    initial_lr = args.initial_lr
    reduction_factor = args.reduction_factor
    num_tries = args.num_tries
    threshold = args.threshold
    
    log_param_dist = args.log_param_dist
    log_grad_dist = args.log_grad_dist
    log_first_epoch = args.log_first_epoch
    find_best_lr(model_type, name=name, dataset=dataset, config_file_name=config_file_name, 
        pool_type=pool_type, initial_lr=initial_lr, reduction_factor=reduction_factor, num_tries=num_tries,
        threshold=threshold, log_param_dist=log_param_dist, log_grad_dist=log_grad_dist, log_first_epoch=log_first_epoch)
