# import tensorboard

import torch.nn as nn
# from layers.pooling_layers import GroupingCombPool2d, GroupingPlusPool2d

def log_distributions(model, writer, iter_number, log_custom_param_dist=False, log_conv_dist=False, log_grad_dist=False,
    custom_modules=None, custom_module_name='custom_module'):

    # if custom_modules is None:
    #     custom_modules = [GroupingPlusPool2d, GroupingCombPool2d]
    

    if log_custom_param_dist:
        custom_module_idx = 0
        for param in model.modules():
            if type(param) in custom_modules: 
                if param.weight is not None:
                    parameter = param.weight.cpu().detach().numpy().squeeze()
                    if parameter.size == 1:
                        writer.add_scalar('{}{}_weight'.format(custom_module_name, custom_module_idx), parameter.item(), iter_number)
                    else:
                        writer.add_histogram('{}{}_weight'.format(custom_module_name, custom_module_idx), parameter, iter_number)
                custom_module_idx += 1
    if log_conv_dist:
        conv_idx = 0
        for param in model.modules():
            if type(param) == nn.Conv2d:  
                weight = param.weight.cpu().detach().numpy().squeeze()
                writer.add_histogram('conv{}_weight'.format(conv_idx), weight, iter_number)
                if param.bias is not None:
                    bias = param.bias.cpu().detach().numpy().squeeze()
                    writer.add_histogram('conv{}_bias'.format(conv_idx), bias, iter_number)
                conv_idx += 1

    if log_grad_dist:

        if log_custom_param_dist:
            custom_module_idx = 0
            for param in model.modules():
                if type(param) in custom_modules: 
                    if param.weight is not None:
                        parameter = param.weight.grad.cpu().detach().numpy().squeeze()
                        if parameter.size == 1:
                            writer.add_scalar('{}{}_weight_grad'.format(custom_module_name, custom_module_idx), parameter.item(), iter_number)
                        else:
                            writer.add_histogram('{}{}_weight_grad'.format(custom_module_name, custom_module_idx), parameter, iter_number)
                    custom_module_idx += 1

        conv_idx = 0
        for param in model.modules():
            if type(param) == nn.Conv2d:
                weight_grad = param.weight.grad.cpu().detach().numpy().squeeze()
                writer.add_histogram('conv{}_weight_grad'.format(conv_idx), weight_grad, iter_number)
                if param.bias is not None:
                    bias_grad = param.bias.grad.cpu().detach().numpy().squeeze()
                    writer.add_histogram('conv{}_bias_grad'.format(conv_idx), bias_grad, iter_number)
                conv_idx += 1