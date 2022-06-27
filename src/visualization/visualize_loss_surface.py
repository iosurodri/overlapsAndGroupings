import torch
import torch.nn as nn

import numpy as np
from src.model_tools.evaluate import evaluate
import matplotlib.pyplot as plt

def get_params_and_directions(model, excluded_parameter_types=None):
    # excluded_parameter_types = [nn.BatchNorm2d]

    # Get the parameters of the model:
    directions1 = []
    directions2 = []
    parameters = []
    modules = model.modules()
    prev_module = next(modules) # Get rid of Linear, since modules is a (consumable) iterable
    for module in modules:
        if ((type(module) != nn.Sequential)) and (len(module._modules) == 0):# or (type(module) in custom_capsule_module_types):
            for param in module.parameters():
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
                    new_value = parameters[param_idx] + alpha * directions1[param_idx] + beta * directions2[param_idx]
                    param.copy_(new_value)
                    param_idx += 1
    return model


def evaluate_loss_surface(model, parameters, directions1, directions2, test_loader, criterion, alphas=np.arange(-1, 1, 0.1), betas=np.arange(-1, 1, 0.1)):

    acc_surface = []
    loss_surface = []
    for alpha in alphas:
        acc_row = []
        loss_row = []
        for beta in betas:
            model = set_model_parameters(model, parameters, directions1, directions2, alpha=alpha, beta=beta)
            accuracy, loss = evaluate(model, criterion, test_loader=test_loader)
            acc_row.append(accuracy)
            loss_row.append(loss)
        acc_surface.append(acc_row)
        loss_surface.append(loss_row)
    return loss_surface, acc_surface


def visualize_loss_surface(model, test_loader, criterion, excluded_parameter_types=None):

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