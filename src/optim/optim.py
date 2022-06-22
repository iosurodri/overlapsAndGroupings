import torch.nn as nn

def get_param_groups(model, custom_module_types, custom_capsule_module_types, completeness_check=False):
    # TODO: Check for edge cases
    modules = model.modules()
    common_modules = nn.ModuleList()
    custom_modules = nn.ModuleList()
    prev_module = None
    module = next(modules)
    for module in modules:
        # TODO: len(module._modules) should indicate if module contains other modules, but there can exist edge cases
        if ((type(module) != nn.Sequential) and (len(module._modules) == 0) or (type(module) in custom_capsule_module_types)) and (type(prev_module) not in custom_capsule_module_types):
            if type(module) in custom_module_types:
                if hasattr(module, 'parameters'):
                    custom_modules.append(module)
            else:
                if hasattr(module, 'parameters'):
                    common_modules.append(module)
        prev_module = module
    common_parameters = common_modules.parameters()
    custom_parameters = custom_modules.parameters()
    # TODO: Add condition to check that all parameters are present (just once) either in custom_modules or common_modules
    # if completeness_check:
        # This condition would be slow, so it should only be used with debugging purposes
    if completeness_check:
        common_parameter_list = list(common_parameters)
        custom_parameter_list = list(custom_parameters)
        for param in model.parameters():
            param_in_common_param_list = False
            param_in_custom_param_list = False
            i = 0
            while i < len(common_parameter_list) and param is not common_parameter_list[i]:
                i += 1
            param_in_common_param_list = i < len(common_parameter_list)
            i = 0
            while i < len(custom_parameter_list) and param is not custom_parameter_list[i]:
                i += 1
            param_in_custom_param_list = i < len(custom_parameter_list)
            if (not param_in_common_param_list) and (not param_in_custom_param_list):
                raise Exception('FAIL: Not all parameters are being optimized')
        raise Exception('SUCCESS: All parameters are being optimized!!')
    return common_parameters, custom_parameters