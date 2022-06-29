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
                new_direction1_weight = torch.rand(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                new_direction2_weight = torch.rand(weight.data.shape, device=weight.data.device, dtype=weight.data.dtype)
                # Reshape weight filters to vector form:
                weight = torch.reshape(weight, [weight.shape[0], weight.shape[1] * weight.shape[2] * weight.shape[3]])
                if module.bias is not None:
                    ### bias=True: The norm of each filter is the norm of the values of the filter's weight, and its bias:
                    bias = module.bias.clone().detach()
                    parameters.append(bias)
                    # Generate random direction for bias term of the convolution layer:
                    new_direction1_bias = torch.rand(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
                    new_direction2_bias = torch.rand(bias.data.shape, device=bias.data.device, dtype=bias.data.dtype)
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
                pass
            else:
                for param in module.parameters():
                    if param is not None:
                        parameters.append(param.clone().detach())
                        if type(module) in excluded_parameter_types:
                            new_direction1 = param.data.new_zeros(param.data.shape)
                            new_direction2 = param.data.new_zeros(param.data.shape)
                        else:
                            new_direction1 = torch.rand(param.data.shape, device=param.data.device, dtype=param.data.dtype)
                            new_direction2 = torch.rand(param.data.shape, device=param.data.device, dtype=param.data.dtype)
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
    return loss_surface, acc_surface


def visualize_loss_surface(model, test_loader, criterion=torch.nn.CrossEntropyLoss(), excluded_parameter_types=[torch.nn.BatchNorm2d]):

    alphas = np.arange(-1, 1, 0.1)
    betas = np.arange(-1, 1, 0.1)

    parameters, directions1, directions2 = get_params_and_directions(model, excluded_parameter_types)
    loss_surface, acc_surface = evaluate_loss_surface(model, parameters, directions1, directions2, test_loader, criterion, alphas=alphas, betas=betas)

    # Generate visualization using PyPlot   
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    loss_surface_np = np.array(loss_surface)
    cp = ax.contour(alphas, betas, loss_surface_np)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('\u03B1')
    ax.set_ylabel('\u03B2')
    plt.show()
    fig.savefig(os.path.join(PATH_PLOTS, 'contour_plot.pdf'), bbox_inches="tight")


    # # custom_capsule_module_types: Includes a list of types which encapsulate parameterized modules. This information
    # #   is used for making sure that the parameters of the child module are not stored twice.

    # center_point = []
    
    # # Get the parameters of the model:
    # modules = model.modules()
    # prev_module = next(modules) # Get rid of Linear, since modules is a (consumable) iterable
    # for module in modules:
    #     if ((type(module) != nn.Sequential)) and (len(module._modules) == 0) or (type(module) in custom_capsule_module_types):
    #         if type(module) in 


    # common_modules = nn.ModuleList()
    # custom_modules = nn.ModuleList()
    # prev_module = None
    # module = next(modules)
    # for module in modules:
    #     # TODO: len(module._modules) should indicate if module contains other modules, but there can exist edge cases
    #     if ((type(module) != nn.Sequential) and (len(module._modules) == 0) or (type(module) in custom_capsule_module_types)) and (type(prev_module) not in custom_capsule_module_types):
    #         if type(module) in custom_module_types:
    #             if hasattr(module, 'parameters'):
    #                 custom_modules.append(module)
    #         else:
    #             if hasattr(module, 'parameters'):
    #                 common_modules.append(module)
    #     prev_module = module
    # common_parameters = common_modules.parameters()
    # custom_parameters = custom_modules.parameters()
    # # TODO: Add condition to check that all parameters are present (just once) either in custom_modules or common_modules
    # # if completeness_check:
    #     # This condition would be slow, so it should only be used with debugging purposes
    # if completeness_check:
    #     common_parameter_list = list(common_parameters)
    #     custom_parameter_list = list(custom_parameters)
    #     for param in model.parameters():
    #         param_in_common_param_list = False
    #         param_in_custom_param_list = False
    #         i = 0
    #         while i < len(common_parameter_list) and param is not common_parameter_list[i]:
    #             i += 1
    #         param_in_common_param_list = i < len(common_parameter_list)
    #         i = 0
    #         while i < len(custom_parameter_list) and param is not custom_parameter_list[i]:
    #             i += 1
    #         param_in_custom_param_list = i < len(custom_parameter_list)
    #         if (not param_in_common_param_list) and (not param_in_custom_param_list):
    #             raise Exception('FAIL: Not all parameters are being optimized')
    #     raise Exception('SUCCESS: All parameters are being optimized!!')
    # return common_parameters, custom_parameters