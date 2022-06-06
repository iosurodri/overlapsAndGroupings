import torch.nn as nn

def get_param_groups(model, custom_module_types, completeness_check=False):
    # TODO: Check for edge cases
    modules = model.modules()
    common_modules = nn.ModuleList()
    custom_modules = nn.ModuleList()
    module = next(modules)
    for module in modules:
        # TODO: len(module._modules) should indicate if module contains other modules, but there can exist edge cases
        if (type(module) != nn.Sequential) and (len(module._modules) == 0):
            if type(module) in custom_module_types:
                if hasattr(module, 'parameters'):
                    custom_modules.append(module)
            else:
                if hasattr(module, 'parameters'):
                    common_modules.append(module)
    common_parameters = common_modules.parameters()
    custom_parameters = custom_modules.parameters()
    # TODO: Add condition to check that all parameters are present (just once) either in custom_modules or common_modules
    # if completeness_check:
        # This condition would be slow, so it should only be used with debugging purposes
    return common_parameters, custom_parameters