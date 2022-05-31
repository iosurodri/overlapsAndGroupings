import torch.nn as nn

def get_param_groups(model, custom_module_types):
    modules = model.modules()
    common_modules = nn.ModuleList()
    custom_modules = nn.ModuleList()
    module = next(modules)
    for module in modules:
        if type(module) != nn.Sequential:
            if type(module) in custom_module_types:
                if hasattr(module, 'parameters'):
                    custom_modules.append(module)
            else:
                if hasattr(module, 'parameters'):
                    common_modules.append(module)
    common_parameters = common_modules.parameters()
    custom_parameters = custom_modules.parameters()
    return common_parameters, custom_parameters