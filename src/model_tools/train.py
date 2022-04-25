import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Auxiliary functions:
from math import ceil, floor

from tqdm import tqdm
from src.data.log_generation import log_distributions
from src.model_tools.save_model import save_model
from src.visualization.visualize_distributions import visualize_heatmap, visualize_hist
from src.functions.loss_functions import SupervisedCrossEntropyLoss
from src.layers.pooling_layers import GroupingPlusPool2d, GroupingCombPool2d


# DEBUG:
from src.data.log_generation import log_activations

PATH_ROOT = os.path.join('..', '..', 'reports')


def train(name, model, optimizer, criterion, train_loader, scheduler=None, train_proportion=1, batch_size=128,
          val_loader=None, num_epochs=20, using_tensorboard=True, save_checkpoints=False, log_param_dist=False, log_grad_dist=False,
          log_first_epoch=False):
    # 0. Prepare auxiliary functionality:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if using_tensorboard:
        try:
            tensorboard_folder = os.path.join(PATH_ROOT, 'runs')
            os.mkdir(tensorboard_folder)
        except:
            pass
        try:
            tensorboard_folder = os.path.join(tensorboard_folder, name)
            os.mkdir(tensorboard_folder)
        except:
            pass
        writer = SummaryWriter(log_dir=tensorboard_folder)

    logs_per_epoch = 1  # Number of logs to be saved by epoch:
    # ToDo: Check that train_loader.dataset takes into account that train_loader doesn't contain all files from dataset
    #  (some are in validation)

    # 1. Training loop:
    # Log information:
    train_acc = []  # Lists for storing accuracy during training
    val_acc = []
    train_loss = []
    val_loss = []
    # Train loop

    # Logic for saving checkpoints:
    if save_checkpoints:
        first_checkpoint = True
        checkpoint_patience = 0
        checkpoint_counter = 0
        best_score = float('inf')
        checkpoint_delta = 0.1  # Another parameter

    # First log of parameters:
    if using_tensorboard:
        if log_param_dist:
            pool_idx = 0
            for param in model.children():
                if type(param) in (GroupingPlusPool2d, GroupingCombPool2d): 
                    if param.weight is not None:
                        parameter = param.weight.cpu().detach().numpy().squeeze()
                        if parameter.size == 1:
                            writer.add_scalar('pool{}_weight'.format(pool_idx), parameter.item(), 0)
                        else:
                            writer.add_histogram('pool{}_weight'.format(pool_idx), parameter, 0)
                    pool_idx += 1

    for epoch in tqdm(range(num_epochs), unit='epochs'):
        running_loss = 0.0
        count_evaluated = 0
        count_correct = 0
        logs_generated = 0

        num_training_samples = len(train_loader.dataset)
        num_batches = ceil((num_training_samples * train_proportion) / batch_size)
        iters_per_log = floor(num_batches / logs_per_epoch)

        # # DEBUG:
        # torch.autograd.set_detect_anomaly(True)

        for i, data in enumerate(tqdm(train_loader, unit='batches', leave=False), 0):

            # Get the inputs; data is a list of [inputs, labels]
            model.train()  # Change network to training mode
            inputs, labels = data[0].to(device), data[1].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Get the predictions for the model
            outputs = model(inputs)
            # Compute the loss value and update weights:
            if type(criterion) == SupervisedCrossEntropyLoss:
                loss = criterion(outputs, labels, epoch=epoch)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # Accumulate the loss for the logging step
            # Count the number of samples correctly predicted:
            count_evaluated += inputs.shape[0]
            if type(criterion) == SupervisedCrossEntropyLoss:
                count_correct += torch.sum(labels == torch.max(outputs[0], dim=1)[1])
            else:
                count_correct += torch.sum(labels == torch.max(outputs, dim=1)[1])
            
            # Useful for debugging early vanishing or exploding gradient problems:
            if log_first_epoch:
                log_distributions(model, writer, iter_number=num_batches * epoch + (i + 1), 
                        log_custom_param_dist=log_param_dist, log_conv_dist=log_param_dist, log_grad_dist=log_grad_dist, 
                        custom_modules=[GroupingPlusPool2d, GroupingCombPool2d], custom_module_name='pool')

            # If it's a logging iteration, generate log data:
            if i % iters_per_log == iters_per_log - 1:
                model.eval()
                logs_generated += 1
                print('Training: [%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / (iters_per_log * logs_generated)))
                train_loss.append(running_loss / (iters_per_log * logs_generated))
                # TensorBoard log generation:
                if using_tensorboard:
                    info = {'loss_train': running_loss / (iters_per_log * logs_generated)}
                    for tag, value in info.items():
                        writer.add_scalar(tag, value, num_batches * epoch + (i + 1))
                    # Log the value of the learning rate in this iteration:
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], num_batches * epoch + (i + 1))
                    # Log the distributions of the parameters of the model:
                    log_distributions(model, writer, iter_number=num_batches * epoch + (i + 1), 
                        log_custom_param_dist=log_param_dist, log_conv_dist=log_param_dist, log_grad_dist=log_grad_dist, 
                        custom_modules=[GroupingPlusPool2d, GroupingCombPool2d], custom_module_name='pool')

        log_first_epoch = False  # Ensure that logging after each batch will only occur on first epoch

        # Validation phase (after each epoch, although could be changed):
        # Evaluate the results from the previous epoch:
        train_acc.append(float(count_correct) / count_evaluated)
        running_loss_val = 0.0
        count_evaluated = 0
        count_correct = 0
        model.eval()  # Change to evaluation mode
        with torch.no_grad():
            for i_val, data_val in enumerate(val_loader, 0):
                inputs_val, labels_val = data_val[0].to(device), data_val[1].to(device)
                outputs_val = model(inputs_val)
                loss = criterion(outputs_val, labels_val)
                running_loss_val += loss.item()
                count_evaluated += inputs_val.shape[0]
                count_correct += torch.sum(labels_val == torch.max(outputs_val, dim=1)[1])
            # Log validation data:
            val_loss.append(running_loss_val / (i_val + 1))
            acc_val = float(count_correct) / count_evaluated
            print('Validation: epoch %d - acc: %.3f' %
                    (epoch + 1, acc_val))
            print('Validation: epoch %d - loss: %.3f' % (epoch + 1, running_loss_val / (i_val + 1)))
            val_acc.append(acc_val)
            if using_tensorboard:
                info = {'acc_val': acc_val, 'loss_val': running_loss_val / (i_val + 1)}
                for tag, value in info.items():
                    writer.add_scalar(tag, value, epoch)
        # Logic for saving best model up to date (according to validation loss)
        if save_checkpoints:
            checkpoint_counter += 1
            if first_checkpoint or (checkpoint_counter > checkpoint_patience):
                first_checkpoint = False
                current_loss = running_loss_val / (i_val + 1)
                if current_loss < (best_score - checkpoint_delta):
                    checkpoint_counter = 0
                    best_score = current_loss
                    save_model(model, name + '_checkpoint')
        # Update scheduler if necessary:
        if scheduler is not None:
            # scheduler.step()
            # DEBUG: Using reduce on plateau:
            if type(scheduler) == ReduceLROnPlateau:
                scheduler.step(running_loss_val / (i_val + 1))
            else:
                scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        # DEBUG:
        # log_activations(model, writer, next(iter(val_loader))[0].to(device), 1)


    return model, train_loss, train_acc, val_loss, val_acc
