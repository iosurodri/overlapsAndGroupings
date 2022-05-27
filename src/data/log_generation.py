# import tensorboard

import torch.nn as nn
# from layers.pooling_layers import GroupingCombPool2d, GroupingPlusPool2d

def log_distribution(writer, param, iter_number, param_type_name, parameter_idx, log_grad_dist=False):
    if hasattr(param, 'weight') and param.weight is not None:
        parameter = param.weight.cpu().detach().numpy().squeeze()
        if parameter.size == 1:
            writer.add_scalar('{}{}_weight'.format(param_type_name, parameter_idx), parameter.item(), iter_number)
        else:
            writer.add_histogram('{}{}_weight'.format(param_type_name, parameter_idx), parameter, iter_number)
        if log_grad_dist:
            parameter = param.weight.grad.cpu().detach().numpy().squeeze()
            if parameter.size == 1:
                writer.add_scalar('{}{}_weight_grad'.format(param_type_name, parameter_idx), parameter.item(), iter_number)
            else:
                writer.add_histogram('{}{}_weight_grad'.format(param_type_name, parameter_idx), parameter, iter_number)
    if hasattr(param, 'bias') and param.bias is not None:
        parameter = param.bias.cpu().detach().numpy().squeeze()
        if parameter.size == 1:
            writer.add_scalar('{}{}_bias'.format(param_type_name, parameter_idx), parameter.item(), iter_number)
        else:
            writer.add_histogram('{}{}_bias'.format(param_type_name, parameter_idx), parameter, iter_number)
        if log_grad_dist:
            parameter = param.bias.grad.cpu().detach().numpy().squeeze()
            if parameter.size == 1:
                writer.add_scalar('{}{}_bias_grad'.format(param_type_name, parameter_idx), parameter.item(), iter_number)
            else:
                writer.add_histogram('{}{}_bias_grad'.format(param_type_name, parameter_idx), parameter, iter_number)


def log_distributions(model, writer, iter_number, log_custom_param_dist=False, log_conv_dist=True, log_linear_dist=True, log_grad_dist=False,
    custom_modules=None, custom_module_name='custom_module'):

    # if custom_modules is None:
    #     custom_modules = [GroupingPlusPool2d, GroupingCombPool2d]
    
    custom_module_idx = 0
    conv_idx = 0
    linear_idx = 0
    for param in model.modules():
        if log_custom_param_dist:
            if type(param) in custom_modules: 
                log_distribution(writer, param, iter_number, custom_module_name, custom_module_idx, log_grad_dist)
                custom_module_idx += 1
        if log_conv_dist:
            if type(param) == nn.Conv2d:
                log_distribution(writer, param, iter_number, 'conv', conv_idx, log_grad_dist) 
                conv_idx += 1
        if log_linear_dist:
            if type(param) == nn.Linear:
                log_distribution(writer, param, iter_number, 'linear', linear_idx, log_grad_dist)
                linear_idx += 1

   
def log_activations(model, writer, sample_batch, iter_number):

    wrapping_modules = [
        nn.Sequential, nn.ModuleList, nn.ModuleDict
    ]

    output = sample_batch
    writer.add_histogram('internal_activations', output.detach().cpu().numpy().squeeze(), iter_number)
    i = 0
    model_skipped = False
    for module_name, module in model.named_modules():
        if not model_skipped:
            model_skipped = True
        else:
            if type(module) not in wrapping_modules:
                output = module(output)
                writer.add_histogram('internal_activations', output.detach().cpu().numpy().squeeze(), iter_number + i)
                i += 1
