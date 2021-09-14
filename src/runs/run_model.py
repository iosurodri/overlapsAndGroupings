# PATH workings:
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset, load_dataset_partition
from src.data.save_results import log_eval_metrics
from src.data.MetricsMeansDesviation import parseMetricsFromFile,calculateMetricStats
# Trainable models:
from src.models.resnet import ResNet
from src.models.resnet import load_resnet
from src.models.dense_net import BigDenseNet
from src.models.dense_net import load_pretrained_densenet

# Model interaction:
from src.model_tools.train import train
from src.model_tools.evaluate import get_prediction_metrics
from src.model_tools.save_model import save_model
from src.model_tools.load_model import load_model
import src.tools.lr_learner as lr_learner

# Auxiliar modules
from src.functions.aggregation_functions import choose_aggregation
from src.functions.focal_loss import FocalLoss
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
    CLI.add_argument("model_type", nargs=1, type=str, help='Type of network to run. Options are: "dense" for '
                                                           'BigDenseNet')
    CLI.add_argument("--dataset", nargs="?", type=str, default="COVID", help='Dataset to be used for training. Options'
                                                                             'are "COVID" for COVID dataset; "COVID2" '
                                                                             'for ASINTOMATICO/INGRESO classes of COVID'
                                                                             'dataset; "COVIDGR" for Granada dataset. '
                                                                             'Defaults to "COVID".')
    CLI.add_argument("--name", nargs="?", type=str, help='Name for the generated files. If none, a name based on the '
                                                         'current date and time will be used instead')
    CLI.add_argument("--pool", nargs="*", type=str, default=None, help="Functions to be used for the pooling layer.")
    CLI.add_argument("--activation", nargs="?", type=str, default=None,
                     help="Activation function to be used. Options: 'relu' for ReLU, 'tanh' for hyperbolic tangential")
    CLI.add_argument("--optimizer", nargs="?", type=str, default=None,
                     help="Optimizer to be used for the model parameters tuning. Options: 'sgd' for Stochastic "
                          "Gradient Descent")
    CLI.add_argument("--loss", nargs="?", type=str, default='cross_entropy',
                     help="Loss function to be used for evaluating predictions. Options: 'cross_entropy' for "
                          "CrossEntropyLoss; 'focal' for FocalLoss")
    CLI.add_argument("--lr", nargs="?", type=float, default=None,
                     help="Value of the optimizer's learning rate")
    CLI.add_argument("--wd", nargs="?", type=float, default=None,
                     help="Value of the optimizer's weight decay.")
    CLI.add_argument("--dr", nargs="?", type=float, default=None,
                     help="Value of the drop rate.")
    CLI.add_argument("--num_layers", nargs="?", type=int, default=None, help="Number of model parameters")
    CLI.add_argument("--classifier_layers", nargs="?", type=int, default=2, help="""Number of layers for model 
        classifier. Valid number of layers are 1 or 2.""")
    CLI.add_argument("--pretrained", nargs="?", type=bool, default=False, help="""Indicates whether we will apply 
        transfer learning to the proposed model.""")
    CLI.add_argument("--partitioned", nargs="?", type=bool, default=False, help="""Indicates whether we will train 
        the model with the whole dataset of with its partitioned version.""")
    CLI.add_argument("--weight_classes", nargs="?", type=bool, default=False, help="""Indicates whether different 
        classes are given different importance when computing loss.""")
    CLI.add_argument("--save_checkpoints", nargs="?", type=bool, default=False, help="""Indicates whether we will save
        the best version of the model obtained during training according to val loss, as well as the final model.""")
    CLI.add_argument("--CrossValidation", nargs="?", type=bool, default=False, help="""Indicates that the model will be
        evaluated by Crossvalidation performance.""")
    CLI.add_argument("--poolType", nargs="?", type=str, default=None, help="""Indicates that the model will be
            aggregate by Gated Pooling""")
    return CLI.parse_args()


def full_process(model_class, pool_aggrs=['mean'], name=None, config_file_name='default_parameters.json',
                 dataset='COVIDGR', optimizer=None, learning_rate=None, weight_decay=None,
                 drop_rate=None, num_layers=None, classifier_layers=1, pretrained=False, partitioned=False,
                 weight_classes=False, loss_function='cross_entropy', save_checkpoints=False, colour=True,
                 crossValidation = False,NumValidations=5,NumParts=5,poolType=None):

    # If no name is specified for referring to the current experiment, we generate one based on the date and hour:
    if name is None:
        date = datetime.datetime.now()
        name = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '__' + str(date.hour) + '_' + str(
            date.minute)
    # If a GPU is available, we will work with it:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Choose the pool functions to be used:
    # ToDo: Debugging -> Change
    if pool_aggrs is None:
        pool_functions = None
    else:
        pool_functions = []
        for pool_aggr in pool_aggrs:
            pool_functions.append(choose_aggregation(pool_aggr))

    # info_data will contain the hyperparameters used for training our model:
    info_data = {}

    with open(os.path.join(CONFIG_PATH, config_file_name)) as config_file:
        config_data = json.load(config_file)
        # Train loop configuration:
        train_params = config_data['train_params']
        num_epochs = train_params['num_epochs']
        batch_size = train_params['batch_size']

        dataset_params = config_data['dataset_params'][dataset]
        num_parts = dataset_params['num_parts']
        num_classes = dataset_params['num_classes']
        train_proportion = dataset_params['train_proportion']
        num_workers = dataset_params['num_workers']
        model_params = config_data['model_params'][model_type]
        if model_class == BigDenseNet:
            # Prepare info_data for storing metadata about the parameters of the model to be saved:
            info_data['network_name'] = 'dense'
            info_data['dataset_name'] = dataset

            info_data['input_size'] = dataset_params['input_size']
            info_data['num_classes'] = num_classes
            info_data['batch_size'] = batch_size
            info_data['growth_rate'] = model_params['growth_rate']
            info_data['memory_efficient'] = model_params['memory_efficient']
            info_data['pool_aggregations'] = pool_aggrs
            info_data['pool_learning_method'] = model_params['pool_learning_method']
            if drop_rate is None:
                drop_rate = model_params['drop_rate']
        elif model_class == ResNet:
            info_data['network_name'] = 'resnet'
            info_data['dataset_name'] = dataset

            info_data['input_size'] = dataset_params['input_size']
            info_data['num_classes'] = num_classes
            info_data['batch_size'] = batch_size
        if optimizer is None:
            optimizer = model_params['optimizer']
        if learning_rate is None:
            learning_rate = model_params['learning_rate']
        if weight_decay is None:
            weight_decay = model_params['weight_decay']
        if num_layers is None:
            num_layers = model_params['num_layers']
        if classifier_layers is None:
            classifier_layers = model_params['classifier_layers']
        info_data['num_layers'] = num_layers
        info_data['classifier_layers'] = classifier_layers
        if colour:
            info_data['in_channels'] = 3
        else:
            info_data['in_channels'] = 1
        momentum = model_params['momentum']
        if poolType is not None:
            info_data['pool_type'] = poolType


    if crossValidation:
        for i in range(1,NumValidations+1):
            for j in range (0,NumParts):
                # 1. Data loading:
                # ToDo: Cargar todos los dataloaders:
                train_dataloaders = []
                if partitioned:
                    for partition_idx in range(num_parts):
                        train_dataloaders.append(load_dataset_partition(partition_idx + 1, dataset, batch_size,
                                                                        num_workers=num_workers, pin_memory=True,
                                                                        colour=colour,NumValidation=i,NumPart=j))
                else:
                    train_dataloader = load_dataset(dataset, batch_size, type='train', num_workers=num_workers, pin_memory=True,
                                                    colour=colour,NumValidation=i,NumPart=j)
                    train_dataloaders.append(train_dataloader)
                val_dataloader = load_dataset(dataset, batch_size, type='val', num_workers=1, pin_memory=True, colour=colour,NumValidation=i,NumPart=j)
                test_dataloader = load_dataset(dataset, batch_size, type='test', num_workers=num_workers, pin_memory=True,
                                               colour=colour,NumValidation=i,NumPart=j)

                # 2. Model initialization:
                if model_class == BigDenseNet:
                    if pretrained:
                        model = load_pretrained_densenet(num_layers=num_layers, bn_size=4, drop_rate=drop_rate,
                                                         num_classes=num_classes,poolType=poolType,
                                                         memory_efficient=model_params['memory_efficient'],
                                                         pool_learning_method=model_params['pool_learning_method'],
                                                         pool_functions=pool_functions, classifier_layers=classifier_layers)
                    else:
                        model = model_class(growth_rate=model_params['growth_rate'], num_layers=num_layers,
                                            bn_size=4, drop_rate=drop_rate, num_classes=num_classes,
                                            memory_efficient=model_params['memory_efficient'],
                                            pool_learning_method=model_params['pool_learning_method'],
                                            pool_functions=pool_functions, in_channels=3,
                                            classifier_layers=classifier_layers)  # DEBUG: classifier_layers=2

                    # DEBUG: in_channels = 3 (?)
                elif model_class == ResNet:
                    model = load_resnet(num_layers, num_classes=num_classes, classifier_layers=classifier_layers,
                                        pretrained=pretrained)
                else:
                    print('Non implemented yet.')
                model.to(device)

                # 3. Optimization method:
                # Optimizer initialization (SGD: Stochastic Gradient Descent):
                if pretrained:
                    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
                else:
                    trainable_parameters = model.parameters()

                if optimizer == 'sgd':
                    # We pass only the non frozen Parameters to the optimizer:
                    realOptimizer = optim.SGD(trainable_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                elif optimizer == 'adam':
                    # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
                    # optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
                    realOptimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
                else:
                    raise Exception('Compatibility with the given optimizer has not been implemented yet')

                # Scheduler: On plateau
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(realOptimizer, factor=0.5, patience=5, threshold=0.0001,
                                                                 cooldown=0,
                                                                 min_lr=0.00001)

                # scheduler_milestones = [round(num_epochs * 0.5), round(num_epochs * 0.75)]
                # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones,
                #                                            gamma=model_params[model_type]['scheduler_factor'])

                if loss_function == 'cross_entropy':
                    if weight_classes:
                        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5509, 5.408]).to(device))
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                elif loss_function == 'focal':
                    # ToDo: DEBUG -> Ensure that the weights are being assigned to the correct class:
                    criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]).to(device),
                                          gamma=2)  # Parameters as provided in original paper
                else:
                    raise Exception('Invalid loss function provided. Available loss functions are "cross_entropy" and "focal"')

                # Set the loss function:

                model, train_loss, train_acc, val_loss, val_acc = train(name+'CV'+str(i)+'_'+str(j), model, realOptimizer, criterion, train_dataloaders,
                                                                        scheduler=scheduler, train_proportion=train_proportion,
                                                                        batch_size=batch_size, val_loader=val_dataloader,
                                                                        num_epochs=num_epochs, using_tensorboard=True,
                                                                        save_checkpoints=save_checkpoints)

                metrics = get_prediction_metrics(model, device, test_dataloader, verbose=False)
                log_eval_metrics(name+'CV'+str(i)+'_'+str(j), metrics,True)
                save_model(model, name+'CV'+str(i)+'_'+str(j), info_data)
                if save_checkpoints:
                    model = load_model(os.path.join(MODELS_PATH, name+'CV'+str(i)+'_'+str(j) + '_checkpoint'), model_type=model_type,
                                       info_data=info_data).to(device)
                    metrics_best = get_prediction_metrics(model, device=device, test_loader=test_dataloader, verbose=False)
                    resultpath=log_eval_metrics(name+'CV'+str(i)+'_'+str(j) + '_best', metrics_best,True)

        parseMetricsFromFile(resultpath)
    else:
        # 1. Data loading:
        # ToDo: Cargar todos los dataloaders:
        train_dataloaders = []
        if partitioned:
            for partition_idx in range(num_parts):
                train_dataloaders.append(load_dataset_partition(partition_idx+1, dataset, batch_size,
                                                                num_workers=num_workers, pin_memory=True, colour=colour))
        else:
            train_dataloader = load_dataset(dataset, batch_size, type='train', num_workers=num_workers, pin_memory=True,
                                            colour=colour)
            train_dataloaders.append(train_dataloader)
        val_dataloader = load_dataset(dataset, batch_size, type='val', num_workers=1, pin_memory=True, colour=colour)
        test_dataloader = load_dataset(dataset, batch_size, type='test', num_workers=num_workers, pin_memory=True,
                                       colour=colour)

        # 2. Model initialization:
        if model_class == BigDenseNet:
            if pretrained:
                model = load_pretrained_densenet(num_layers=num_layers, bn_size=4, drop_rate=drop_rate,
                                                 num_classes=num_classes, memory_efficient=model_params['memory_efficient'],
                                                 pool_learning_method=model_params['pool_learning_method'],
                                                 pool_functions=pool_functions, classifier_layers=classifier_layers,poolType=poolType)
            else:
                model = model_class(growth_rate=model_params['growth_rate'], num_layers=num_layers,
                                    bn_size=4, drop_rate=drop_rate, num_classes=num_classes,
                                    memory_efficient=model_params['memory_efficient'],
                                    pool_learning_method=model_params['pool_learning_method'],
                                    pool_functions=pool_functions, in_channels=3,
                                    classifier_layers=classifier_layers)  # DEBUG: classifier_layers=2

            # DEBUG: in_channels = 3 (?)
        elif model_class == ResNet:
            model = load_resnet(num_layers, num_classes=num_classes, classifier_layers=classifier_layers, pretrained=pretrained)
        else:
            print('Non implemented yet.')
        model.to(device)

        # 3. Optimization method:
        # Optimizer initialization (SGD: Stochastic Gradient Descent):
        if pretrained:
            trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
        else:
            trainable_parameters = model.parameters()

        if optimizer == 'sgd':
            # We pass only the non frozen Parameters to the optimizer:
            optimizer = optim.SGD(trainable_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adam':
            # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
            # optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception('Compatibility with the given optimizer has not been implemented yet')

        # Scheduler: On plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=0.0001, cooldown=0,
                                                         min_lr=0.00001)

        # scheduler_milestones = [round(num_epochs * 0.5), round(num_epochs * 0.75)]
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones,
        #                                            gamma=model_params[model_type]['scheduler_factor'])

        if loss_function == 'cross_entropy':
            if weight_classes:
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5509, 5.408]).to(device))
            else:
                criterion = torch.nn.CrossEntropyLoss()
        elif loss_function == 'focal':
            # ToDo: DEBUG -> Ensure that the weights are being assigned to the correct class:
            criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]).to(device), gamma=2)  # Parameters as provided in original paper
        else:
            raise Exception('Invalid loss function provided. Available loss functions are "cross_entropy" and "focal"')

        # Set the loss function:

        model, train_loss, train_acc, val_loss, val_acc = train(name, model, optimizer, criterion, train_dataloaders,
                                                                scheduler=scheduler, train_proportion=train_proportion,
                                                                batch_size=batch_size, val_loader=val_dataloader,
                                                                num_epochs=num_epochs, using_tensorboard=True,
                                                                save_checkpoints=save_checkpoints)

        metrics = get_prediction_metrics(model, device, test_dataloader, verbose=False)
        log_eval_metrics(name, metrics)
        save_model(model, name, info_data)
        if save_checkpoints:
            model = load_model(os.path.join(MODELS_PATH, name + '_checkpoint'), model_type=model_type,
                               info_data=info_data).to(device)
            metrics_best = get_prediction_metrics(model, device=device, test_loader=test_dataloader, verbose=False)
            log_eval_metrics(name + '_best', metrics_best)


if __name__ == '__main__':
    # PREPROCESS of sys.argv for compatibility with gnu parallel:
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    model_type = args.model_type[0]
    dataset = args.dataset
    pool = None
    if model_type == 'dense':
        model_class = BigDenseNet

    elif model_type == 'resnet':
        model_class = ResNet
    else:
        raise Exception('Wrong model_type parameter provided ({}): Options are "ref" for REFNet or "Standard" for '
                        'StandardNet'.format(model_type))
    num_layers = args.num_layers
    classifier_layers = args.classifier_layers
    pool_aggrs = args.pool
    optimizer = args.optimizer
    lr = args.lr
    wd = args.wd
    name = args.name
    dr = args.dr
    pretrained = args.pretrained
    partitioned = args.partitioned
    weight_classes = args.weight_classes
    loss = args.loss
    save_checkpoints = args.save_checkpoints
    crossValidation = args.CrossValidation
    poolType = args.poolType
    full_process(model_class, pool_aggrs, name, dataset=dataset, optimizer=optimizer, learning_rate=lr, weight_decay=wd,
                 drop_rate=dr, num_layers=num_layers, classifier_layers=classifier_layers, pretrained=pretrained,
                 partitioned=partitioned, weight_classes=weight_classes, loss_function=loss,
                 save_checkpoints=save_checkpoints,crossValidation=crossValidation,poolType=poolType)
