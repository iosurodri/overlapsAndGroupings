import torch
import torch.nn as nn

import numpy as np
from src.model_tools.evaluate import evaluate
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
PATH_REPORTS = os.path.join('..', '..', 'reports')

PATH_PLOTS = os.path.join(PATH_REPORTS, 'plots')
try:
    os.mkdir(PATH_PLOTS)
except FileExistsError as e:
    pass

def get_params_and_directions(model, excluded_parameter_types=[nn.BatchNorm2d]):
    """Extracts a copy of the parameters of a given model and generates two random (filter normalized) direction vectors in the feature space:
    If the features of a feature model are seen as a vector F in a dimensional space (defined by all the parameters of all layers of the model),
    two random vectors in the same dimensional space d1 and d2 are sampled from a gaussian distribution. 
    Note: In order to avoid scale invariance issues, the parameters of d1 and d2 corresponding to convolutional or linear layers are normalized
    based on their corresponding filter norms. That is, roughly speaking: d1_i <- (d1_i / ||d1_i||) * ||F_i||
    More info (source): Visualizing the Loss Landscape of Neural Nets (https://arxiv.org/abs/1712.09913)

    Args:
        model (modelType): Model from which parameters are to be copied.
        excluded_parameter_types (list<torch.nn.Module>, optional): List of torch.nn.Module types for which directions are to be ignored. 
            In the original paper, Batch Normalization layer parameters are not considered in d1 and d2 (vectors of zeros are generated for these parameters).
            Other user defined torch.nn.Modules can be included as well.

    Returns:
        list<torch.nn.Tensor>: A copy of all the parameters in "model", returned in the order presented by model.parameters()
        list<torch.nn.Tensor>: First random direction
        list<torch.nn.Tensor>: Second random direction
    """

    # excluded_parameter_types = [nn.BatchNorm2d]

    # Get the parameters of the model:
    directions1 = []
    directions2 = []
    parameters = []
    modules = model.modules()
    _ = next(modules) # Get rid of Linear, since modules is a (consumable) iterable
    for module in modules:
        if ((type(module) != nn.Sequential)) and (len(module._modules) == 0):# or (type(module) in custom_capsule_module_types):
            if type(module) == torch.nn.Conv2d:
                '''Filter normalization: 
                   In order to avoid scale invariance problems, the filters of convolutional layers (and linear layers)
                   must be normalized so that each filter's random direction has norm equal to its filter's norm.
                '''
                weight = module.weight.clone().detach()
                parameters.append(weight)
                # Generate random directions for filters of the convolution layer:                    
                new_direction1_weight = torch.randn(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                new_direction2_weight = torch.randn(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                # Reshape weight filters to vector form:
                weight = torch.reshape(weight, [weight.shape[0], weight.shape[1] * weight.shape[2] * weight.shape[3]])
                if module.bias is not None:
                    ### bias=True: The norm of each filter is the norm of the values of the filter's weight, and its bias:
                    bias = module.bias.clone().detach()
                    parameters.append(bias)
                    # Generate random direction for bias term of the convolution layer:
                    new_direction1_bias = torch.randn(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
                    new_direction2_bias = torch.randn(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
                    # Reshape bias term to be compatible with weigh:
                    bias = bias.unsqueeze(1)
                    # Compute the norm of each filter (taking into account the weight values and the bias term):
                    filter_norm = torch.norm(torch.concat([weight, bias], dim=1), dim=1)
                    # Reshape the random directions taking into account the term for the weight values and bias term and compute their norms:
                    new_direction1 = torch.concat([
                        torch.reshape(new_direction1_weight, weight.shape), new_direction1_bias.unsqueeze(1)
                    ], dim=1)
                    new_direction1_norm = torch.norm(new_direction1, dim=1)
                    new_direction2 = torch.concat([
                        torch.reshape(new_direction2_weight, weight.shape), new_direction2_bias.unsqueeze(1)
                    ], dim=1)
                    new_direction2_norm = torch.norm(new_direction2, dim=1)
                    # Normalize all the random directions to have norm equal to the norm of the filter:
                    directions1.append((new_direction1_weight / new_direction1_norm[:, None, None, None]) * filter_norm[:, None, None, None])
                    directions2.append((new_direction2_weight / new_direction2_norm[:, None, None, None]) * filter_norm[:, None, None, None])
                    directions1.append((new_direction1_bias / new_direction1_norm) * filter_norm)
                    directions2.append((new_direction2_bias / new_direction2_norm) * filter_norm)
                else:
                    ### bias=False: If there is no bias, then the norm of each filter is only affected by its weight:
                    filter_norm = torch.norm(weight, dim=1)
                    new_direction1_norm = torch.norm(torch.reshape(new_direction1_weight, weight.shape), dim=1)
                    new_direction2_norm = torch.norm(torch.reshape(new_direction2_weight, weight.shape), dim=1)
                    # Filter normalize the random directions:
                    directions1.append((new_direction1_weight / new_direction1_norm[:, None, None, None]) * filter_norm[:, None, None, None])  # new_direction1_norm[:, None, None, None] acts as a "broadcastable operation":
                    directions2.append((new_direction2_weight / new_direction1_norm[:, None, None, None]) * filter_norm[:, None, None, None])
            elif type(module) == torch.nn.Linear:
                # TODO: Implement
                weight = module.weight.clone().detach()
                parameters.append(weight)
                # Generate random directions for filters of the convolution layer:                    
                new_direction1_weight = torch.randn(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                new_direction2_weight = torch.randn(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                if module.bias is not None:
                    ### bias=True: The norm of each filter is the norm of the values of the filter's weight, and its bias:
                    bias = module.bias.clone().detach()
                    parameters.append(bias)
                    # Generate random direction for bias term of the convolution layer:
                    new_direction1_bias = torch.randn(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
                    new_direction2_bias = torch.randn(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
                    # Reshape bias term to be compatible with weigh:
                    bias = bias.unsqueeze(1)
                    # Compute the norm of each filter (taking into account the weight values and the bias term):
                    filter_norm = torch.norm(torch.concat([weight, bias], dim=1), dim=1)
                    # Reshape the random directions taking into account the term for the weight values and bias term and compute their norms:
                    new_direction1 = torch.concat([new_direction1_weight, new_direction1_bias.unsqueeze(1)], dim=1)
                    new_direction1_norm = torch.norm(new_direction1, dim=1)
                    new_direction2 = torch.concat([new_direction2_weight, new_direction2_bias.unsqueeze(1)], dim=1)
                    new_direction2_norm = torch.norm(new_direction2, dim=1)
                    # Normalize all the random directions to have norm equal to the norm of the filter:
                    directions1.append((new_direction1_weight / new_direction1_norm[:, None]) * filter_norm[:, None])
                    directions2.append((new_direction2_weight / new_direction2_norm[:, None]) * filter_norm[:, None])
                    directions1.append((new_direction1_bias / new_direction1_norm) * filter_norm)
                    directions2.append((new_direction2_bias / new_direction2_norm) * filter_norm)
                else:
                    ### bias=False: If there is no bias, then the norm of each filter is only affected by its weight:
                    filter_norm = torch.norm(weight, dim=1)
                    new_direction1_norm = torch.norm(new_direction1_weight, dim=1)
                    new_direction2_norm = torch.norm(new_direction2_weight, dim=1)
                    # Filter normalize the random directions:
                    directions1.append((new_direction1_weight / new_direction1_norm[:, None]) * filter_norm[:, None])  # new_direction1_norm[:, None, None, None] acts as a "broadcastable operation":
                    directions2.append((new_direction2_weight / new_direction1_norm[:, None]) * filter_norm[:, None])
            else:
                for param in module.parameters():
                    if param is not None:
                        parameters.append(param.clone().detach())
                        if type(module) in excluded_parameter_types:
                            new_direction1 = param.data.new_zeros(param.data.shape)
                            new_direction2 = param.data.new_zeros(param.data.shape)
                        else:
                            new_direction1 = torch.randn(param.data.shape, device=param.data.device, dtype=param.data.dtype)
                            new_direction2 = torch.randn(param.data.shape, device=param.data.device, dtype=param.data.dtype)
                            # while torch.abs(torch.cosine_similarity(new_direction1, new_direction2)) < 0.5:
                                # Regenerate the second random direction (we want them to be different)
                                # new_direction2 = torch.rand(param.data.shape, device=param.data.device, dtype=param.data.dtype)
                            # TODO: Filterwise normalization:
                            if (type(module) == nn.Conv2d) and (param.dim() == 4):
                                new_direction1 = (new_direction1 / torch.norm(new_direction1, dim=0, keepdim=True)) * torch.norm(param.data, dim=0, keepdim=True)
                                new_direction2 = (new_direction2 / torch.norm(new_direction2, dim=0, keepdim=True)) * torch.norm(param.data, dim=0, keepdim=True)
                            elif type(module) == nn.Linear:
                                new_direction1 = (new_direction1 / torch.norm(new_direction1, dim=0, keepdim=True)) * torch.norm(param.data, dim=0, keepdim=True)
                                new_direction2 = (new_direction2 / torch.norm(new_direction2, dim=0, keepdim=True)) * torch.norm(param.data, dim=0, keepdim=True)
                        directions1.append(new_direction1)  # We could also append something like [type(param), new_direction1]
                        directions2.append(new_direction2)
    return parameters, directions1, directions2


def set_model_parameters(model, parameters, directions1, directions2, alpha=0, beta=0):
    modules = model.modules()
    module = next(modules) # Get rid of Linear, since modules is a (consumable) iterable
    param_idx = 0
    with torch.no_grad():
        for module in modules:
            if ((type(module) != nn.Sequential)) and (len(module._modules) == 0):# or (type(module) in custom_capsule_module_types):
                for param in module.parameters():
                    if param is not None:  # For torch.nn.Conv2d or torch.nn.Linear, if bias=False, param can be None
                        new_value = parameters[param_idx] + alpha * directions1[param_idx] + beta * directions2[param_idx]
                        param.copy_(new_value)
                        param_idx += 1
    return model


def evaluate_loss_surface(model, parameters, directions1, directions2, test_loader, criterion, alphas=np.arange(-1, 1, 0.1), betas=np.arange(-1, 1, 0.1)):
    """Generate numpy arrays representing the cost (and accuracy) surface of a given model, evaluated with parameters:
    model.features = features + (alphas_i * directions1) + (betas_j * directions2)

    Args:
        model (modelType): Model to be evaluated with different parameter configurations.
        parameters (_type_): _description_
        directions1 (_type_): _description_
        directions2 (_type_): _description_
        test_loader (_type_): _description_
        criterion (_type_): _description_
        alphas (_type_, optional): _description_. Defaults to np.arange(-1, 1, 0.1).
        betas (_type_, optional): _description_. Defaults to np.arange(-1, 1, 0.1).

    Returns:
        _type_: _description_
    """
    acc_surface = []
    loss_surface = []
    for alpha in tqdm(alphas, unit='alphas'):
        acc_row = []
        loss_row = []
        for beta in tqdm(betas, unit='betas'):
            model = set_model_parameters(model, parameters, directions1, directions2, alpha=alpha, beta=beta)
            accuracy, loss = evaluate(model, criterion, test_loader=test_loader)
            acc_row.append(accuracy)
            loss_row.append(loss)
        acc_surface.append(acc_row)
        loss_surface.append(loss_row)
    return np.array(loss_surface), np.array(acc_surface)


def visualize_loss_surface(model, test_loader, name, criterion=torch.nn.CrossEntropyLoss(), excluded_parameter_types=[torch.nn.BatchNorm2d]):

    alphas = np.arange(-1, 1, 0.1)
    betas = np.arange(-1, 1, 0.1)

    parameters, directions1, directions2 = get_params_and_directions(model, excluded_parameter_types)
    loss_surface_np, acc_surface_np = evaluate_loss_surface(model, parameters, directions1, directions2, test_loader, criterion, alphas=alphas, betas=betas)
    # Save loss_surface and acc_surface as csv files:
    np.savetext(os.path.join(PATH_PLOTS, 'loss_surface_{}.csv'.format(name)), loss_surface_np, delimiter=",")
    np.savetext(os.path.join(PATH_PLOTS, 'acc_surface_{}.csv'.format(name)), acc_surface_np, delimiter=",")

    # Plot loss surface:
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    cp = ax.contour(alphas, betas, loss_surface_np)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour Plot (Loss)')
    ax.set_xlabel('\u03B1')
    ax.set_ylabel('\u03B2')
    plt.show()
    fig.savefig(os.path.join(PATH_PLOTS, 'loss_surface_{}.pdf'.format(name)), bbox_inches="tight")
    fig.savefig(os.path.join(PATH_PLOTS, 'loss_surface_{}.png'.format(name)), bbox_inches="tight")

    # Plot accuracy surface:
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    cp = ax.contour(alphas, betas, acc_surface_np)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour Plot (Accuracy)')
    ax.set_xlabel('\u03B1')
    ax.set_ylabel('\u03B2')
    plt.show()
    fig.savefig(os.path.join(PATH_PLOTS, 'acc_surface_{}.pdf'.format(name)), bbox_inches="tight")
    fig.savefig(os.path.join(PATH_PLOTS, 'acc_surface_{}.pdf'.format(name)), bbox_inches="tight")