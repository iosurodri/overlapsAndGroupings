import os
import sys
sys.path.append(os.path.join(os.path.realpath('.'), os.pardir, os.pardir))

import torch
import torch.nn.functional as F

import os

import src.functions.aggregation_functions as aggr_funcs
import src.functions.aux_functions as aux_funcs


class OverlapPool2d(torch.nn.Module):

    available_overlaps = {
        # 'product': torch.prod,
        'product': lambda x, dim=-1, keepdim=False: torch.prod(x, keepdim=keepdim, dim=dim),
        # TODO: SEGUIR AQUÍ
        'product_derivative1': lambda x, dim=-1, keepdim=False: torch.prod(x, keepdim=keepdim, dim=dim) + torch.sum(x, keepdim=keepdim, dim=dim),
        'product_derivative1_k': lambda x, dim=-1, keepdim=False: (1 - x.shape[dim]) * torch.prod(x, keepdim=keepdim, dim=dim) + torch.sum(x, keepdim=keepdim, dim=dim),
        'minimum': lambda x, dim=-1, keepdim=False: torch.min(x, dim=dim, keepdim=keepdim)[0],
        'ob': aggr_funcs.ob_overlap,
        'geometric': aggr_funcs.geometric_mean
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, overlap=None, normalization=None, denormalize=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if overlap not in self.available_overlaps.keys():
            raise Exception('Overlap {} unavailable for OverlapPool2d. Must be one of {}'.format(
                overlap, self.available_overlaps.keys()))
        self.overlap = self.available_overlaps[overlap]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for OverlapPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that overlaps defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen overlap:
        output_tensor = self.overlap(output_tensor, dim=-1)
        # 5.-Denormalize output values after applying overlap?
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        return output_tensor


class GroupingPool2d(torch.nn.Module):

    available_groupings = {
        'product': aggr_funcs.product_grouping,
        'minimum': aggr_funcs.minimum_grouping,
        'maximum': lambda x, dim=-1: torch.max(x, dim=dim)[0],
        'maximum_sqrt': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 0.5),
        'maximum_square': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 2),
        'ob': aggr_funcs.ob_grouping,
        'geometric': aggr_funcs.geometric_grouping,
        'u': aggr_funcs.u_grouping
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization=None, denormalize=True, clip_grad=1):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if grouping not in self.available_groupings.keys():
            raise Exception('Grouping {} unavailable for GroupingPool2d. Must be one of {}'.format(
                grouping, self.available_groupings.keys()))
        self.grouping = self.available_groupings[grouping]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for GroupingPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']
        
        # DEBUG: Testing gradient clipping using backward hooks:
        if clip_grad is not None:
            self.clip_grad = clip_grad

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)

        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))

        # # DEBUG: Testing gradient clipping using backward hooks:
        if self.clip_grad and self.training and tensor.requires_grad:
        #     tensor.register_hook(lambda grad: print(grad))
            # tensor.register_hook(lambda grad: print(grad.mean()))
            tensor.register_hook(lambda grad: torch.clamp(grad, min=-torch.quantile(grad, 0.001), max=torch.quantile(grad, 0.001)))
            # tensor.register_hook(lambda grad: torch.clamp(grad, min=-.001, max=.001))
            # tensor.register_hook(lambda grad: print(grad.mean()))
        #     tensor.register_hook(lambda grad: print(grad))
        
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        
        # DEBUG: Testing gradient clipping using backward hooks:
        # if self.clip_grad and self.training and output_tensor.requires_grad:
        #     output_tensor.register_hook(lambda grad: torch.clamp(grad, min=-1, max=1))
        
        # 4.-Compute reduction based on the chosen grouping:
        output_tensor = self.grouping(output_tensor, dim=-1)

        # 5.-Denormalize output values after applying grouping
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)

        return output_tensor


class GroupingPlusPool2d(torch.nn.Module):

    available_groupings = {
        'max_power': aggr_funcs.max_power_grouping,
        'product_power': aggr_funcs.product_power_grouping,
        'geometric_power': aggr_funcs.geometric_power_grouping
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization=None, denormalize=True, weight_mode='single',
                 num_channels=1, initial_pool_exp=None):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        # DEBUG: removing parameter initial_pool_exp:
        if initial_pool_exp is None:
            initial_pool_exp = 1.0
        
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if grouping not in self.available_groupings.keys():
            raise Exception('Grouping {} unavailable for GroupingPlusPool2d. Must be one of {}'.format(
                grouping, self.available_groupings.keys()))
        self.grouping = self.available_groupings[grouping]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for GroupingPlusPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

        if weight_mode == 'single':
            self.weight = torch.nn.Parameter(torch.ones([1, 1, 1, 1, 1]) * initial_pool_exp)
        elif weight_mode == 'channel_wise':
            self.weight = torch.nn.Parameter(torch.ones([1, num_channels, 1, 1, 1]) * initial_pool_exp)
        else:
            raise Exception('Wrong option for weight_mode provided: {}'.format(weight_mode))
        

    def forward(self, tensor):

        # DEBUG: Requirement for p > 0
        square_weight = self.weight * self.weight

        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # If automorphisms are to be used, apply them before applying the grouping (if chosen to do so):
        
        # 4.-Compute reduction based on the chosen grouping:
        output_tensor = self.grouping(output_tensor, square_weight, dim=-1)

        # 5.-Denormalize output values after applying grouping
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)

        return output_tensor

class GroupingCompositionPool2d(torch.nn.Module):

    available_groupings = {
        'product': aggr_funcs.product_grouping,
        'minimum': aggr_funcs.minimum_grouping,
        'maximum': lambda x, dim=-1: torch.max(x, dim=dim)[0],
        'maximum_sqrt': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 0.5),
        'maximum_square': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 2),
        'ob': aggr_funcs.ob_grouping,
        'geometric': aggr_funcs.geometric_grouping,
        'u': aggr_funcs.u_grouping
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_big='max', grouping_list=None, normalization=None, denormalize=True):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if grouping_big not in self.available_groupings.keys():
            raise Exception('Grouping {} unavailable. Must be one of {}'.format(
                grouping_big, self.available_groupings.keys()))
        for grouping in grouping_list:
            if grouping not in self.available_groupings.keys():
                raise Exception('Grouping {} unavailable. Must be one of {}'.format(
                    grouping, self.available_groupings.keys()))        
        self.grouping_big = self.available_groupings[grouping_big]
        self.grouping_list = [self.available_groupings[grouping] for grouping in grouping_list]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for GroupingPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        normalized_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen grouping:

        # Generate an auxiliar tensor the size of normalized_tensor, with as many values for the last dimension as groupings to apply
        output_tensor = tensor.new_zeros([*tensor.shape[:-1], len(self.grouping_list)])
        for idx, grouping in enumerate(self.grouping_list):
            output_tensor[..., idx] = grouping(normalized_tensor, dim=-1) 

        # Aggregate all the obtained outputs (values on the last dimension) by means of self.grouping_big
        output_tensor = self.grouping_big(output_tensor, dim=-1)

        # 5.-Denormalize output values after applying grouping
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)

        return output_tensor


class GroupingCombPool2d(torch.nn.Module):

    available_groupings = {
        'average': torch.mean,  # Not a grouping, but convenient for tests
        'product': aggr_funcs.product_grouping,
        'minimum': aggr_funcs.minimum_grouping,
        'maximum': lambda x, dim=-1: torch.max(x, dim=dim)[0],
        'maximum_sqrt': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 0.5),
        'maximum_square': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 2),
        'ob': aggr_funcs.ob_grouping,
        'geometric': aggr_funcs.geometric_grouping,
        'u': aggr_funcs.u_grouping
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_list=None, normalization=None, denormalize=True, learnable=True):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        for grouping in grouping_list:
            if grouping not in self.available_groupings.keys():
                raise Exception('Grouping {} unavailable. Must be one of {}'.format(
                    grouping, self.available_groupings.keys()))        
        self.grouping_list = [self.available_groupings[grouping] for grouping in grouping_list]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for GroupingPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']
        self.weight = None
        if learnable:
            self.weight = torch.nn.Parameter(torch.ones([len(self.grouping_list)]) * (1 / len(self.grouping_list)))

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        normalized_tensor, normalization_params = self.normalization(tensor)

        if self.weight is None:
            weight = [1 / len(self.grouping_list) for i in range(len(self.grouping_list))]
        else:
            weight = torch.softmax(self.weight, dim=-1)

        # 4.-Compute reduction based on the chosen grouping:
        # Generate an auxiliar tensor the size of normalized_tensor, with as many values for the last dimension as groupings to apply
        output_tensor = tensor.new_zeros(tensor.shape[:-1])
        for idx, grouping in enumerate(self.grouping_list):
            output_tensor += weight[idx] * grouping(normalized_tensor, dim=-1) 

        # 5.-Denormalize output values after applying grouping
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)

        return output_tensor


class TnormPool2d(torch.nn.Module):

    available_tnorms = {
        'minimum': lambda x, dim=-1, keepdim=False: torch.min(x, keepdim=keepdim, dim=dim)[0],
        'product': lambda x, dim=-1, keepdim=False: torch.prod(x, keepdim=keepdim, dim=dim),
        'lukasiewicz': aggr_funcs.lukasiewicz_tnorm,
        'hamacher': aggr_funcs.hamacher_tnorm
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, tnorm=None, normalization=None, denormalize=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if tnorm not in self.available_tnorms.keys():
            raise Exception('Tnorm {} unavailable for TnormPool2d. Must be one of {}'.format(
                tnorm, self.available_tnorms.keys()))
        self.tnorm = self.available_tnorms[tnorm]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for TnormPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that tnorms defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen tnorm:
        output_tensor = self.tnorm(output_tensor, dim=-1)
        # 5.-Denormalize output values after applying tnorm
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        return output_tensor


class TconormPool2d(torch.nn.Module):

    available_tconorms = {
        'maximum': lambda x, dim=-1, keepdim=False: torch.max(x, keepdim=keepdim, dim=dim)[0],
        'prob_sum': aggr_funcs.probabilistic_sum,
        'bounded_sum': aggr_funcs.bounded_sum,
        'hamacher': aggr_funcs.hamacher_tconorm,
        'einstein_sum': aggr_funcs.einstein_sum
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, tconorm=None, normalization=None, denormalize=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if tconorm not in self.available_tconorms.keys():
            raise Exception('Tconorm {} unavailable for TconormPool2d. Must be one of {}'.format(
                tconorm, self.available_tconorms.keys()))
        self.tconorm = self.available_tconorms[tconorm]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for TconormPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that tconorms defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen tconorm:
        output_tensor = self.tconorm(output_tensor, dim=-1)
        # 5.-Denormalize output values after applying tconorm
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        return output_tensor


class UninormPool2d(torch.nn.Module):

    available_uninorms = {
        'min_max': aggr_funcs.uninorm_min_max,
        'product': aggr_funcs.uninorm_product,
        'lukasiewicz': aggr_funcs.uninorm_lukasiewicz,
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, uninorm=None, normalization=None, denormalize=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if uninorm not in self.available_uninorms.keys():
            raise Exception('Uninorm {} unavailable for UninormPool2d. Must be one of {}'.format(
                uninorm, self.available_uninorms.keys()))
        self.uninorm = self.available_uninorms[uninorm]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for UninormPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that uninorms defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen uninorm:
        output_tensor = self.uninorm(output_tensor, dim=-1)
        # 5.-Denormalize output values after applying uninorm
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        return output_tensor


class MDPool2d(torch.nn.Module):


    available_deviations = {
        'test': lambda x, keepdim=False, dim=-1: aggr_funcs.basic_moderate_deviation(x, keepdim=keepdim, dim=dim, m=2),
        'm1_5': lambda x, keepdim=False, dim=-1: aggr_funcs.basic_moderate_deviation(x, keepdim=keepdim, dim=dim, m=1.5),
        'm2_5': lambda x, keepdim=False, dim=-1: aggr_funcs.basic_moderate_deviation(x, keepdim=keepdim, dim=dim, m=2.5),
        'm3': lambda x, keepdim=False, dim=-1: aggr_funcs.basic_moderate_deviation(x, keepdim=keepdim, dim=dim, m=2.5),
    }

    available_normalizations = {
        'linear': {
            'normalization': aux_funcs.linearAb2minusOneOne,
            'denormalization': aux_funcs.linearMinusOneOne2ab
        },
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, deviation=None, normalization=None, denormalize=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        if deviation not in self.available_deviations.keys():
            raise Exception('Deviation {} unavailable for MDPool2d. Must be one of {}'.format(
                deviation, self.available_deviations.keys()))
        self.deviation = self.available_deviations[deviation]
        if normalization not in self.available_normalizations.keys():
            raise Exception('Normalization {} unavailable for MDPool2d. Must be one of {}'.format(
                normalization, self.available_normalizations.keys()))
        self.normalization = self.available_normalizations[normalization]['normalization']
        self.denormalize = denormalize
        self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that deviations defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based on the chosen deviation:
        output_tensor = self.deviation(output_tensor, dim=-1)
        # 5.-Denormalize output values after applying deviation
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        return output_tensor


class ResidualPool2d(torch.nn.Module):

    '''Implements residual connections into pooling layers:
    res_pool(X) = pool(X) + res(X)
    where pool is the pooling function to be performed and res the residual term
    Options for res:
        1) res(X) = x_1 + ... + x_n
        2) res(X) = w_1 * x_1 + ... + w_n * x_1  TODO
    '''

    def __init__(self, pool_layer, kernel_size=2, stride=None, res_type='identity'):
        super().__init__()
        self.pool_layer = pool_layer
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        if res_type == 'identity':
            self.weight = torch.ones([1], dtype=torch.float)
        elif res_type == 'same_convex':
            self.weight = torch.nn.Parameter(torch.zeros([1], dtype=torch.float))
        elif res_type == 'diff_convex':
            self.weight = torch.nn.Parameter(torch.zeros([kernel_size[0] * kernel_size[1]], dtype=torch.float))
        elif res_type == 'same':
            self.weight = torch.nn.Parameter(torch.ones([1], dtype=torch.float))
        elif res_type == 'diff':
            self.weight = torch.nn.Parameter(torch.ones([kernel_size[0] * kernel_size[1]], dtype=torch.float))
        self.res_type = res_type

    def forward(self, tensor):
        # 1) Apply pooling to tensor:
        output = self.pool_layer(tensor)
        # 2) Extract patches from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3) Add residual information:
        if self.res_type == 'identity':
            output = output + torch.sum(tensor, dim=-1)
        elif self.res_type == 'mean':
            output = output + torch.mean(tensor, dim=-1)  # Equivalent to "same" with self.coeff = 1/(kernel_size * kernel_size)
        elif self.res_type == 'same_convex':
            # TODO: Fix get_params
            # new_coeff = torch.softmax(torch.cat([self.coeff, -self.coeff], dim=0), dim=0)
            # output = new_coeff[0] * output + new_coeff[1] * torch.sum(tensor, dim=-1)
            # DANGER: torch.cat is not differentiable
            new_coeff_alpha = torch.exp(self.weight) / (torch.exp(self.weight) + torch.exp(-self.weight))
            new_coeff_beta = torch.exp(-self.weight) / (torch.exp(self.weight) + torch.exp(-self.weight))
            output = new_coeff_alpha * output + new_coeff_beta * torch.sum(tensor, dim=-1)
        elif self.res_type == 'diff_convex':
            new_coeff = torch.softmax(torch.cat([self.coeff, -self.coeff], dim=1), dim=1)
            output = new_coeff[0] * output + new_coeff[1] * torch.sum(tensor, dim=-1)
        elif self.res_type == 'same' or self.res_type == 'diff':
            self.coeff = output + self.coeff * torch.sum(tensor, dim=-1)
        return output


class AttentionPool2d(torch.nn.Module):
    
    available_groupings = {
        'product': aggr_funcs.product_grouping,
        'minimum': aggr_funcs.minimum_grouping,
        'maximum': lambda x, dim=-1: torch.max(x, dim=dim)[0],
        'maximum_sqrt': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 0.5),
        'maximum_square': lambda x: torch.pow(torch.max(x)[0] + 0.000001, 2),
        'ob': aggr_funcs.ob_grouping,
        'geometric': aggr_funcs.geometric_grouping,
        'u': aggr_funcs.u_grouping
    }

    available_normalizations = {
        'min_max': {
            'normalization': aux_funcs.min_max_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'quantile': {
            'normalization': aux_funcs.quantile_normalization,
            'denormalization': aux_funcs.min_max_denormalization
        },
        'sigmoid': {
            'normalization': lambda x: (torch.sigmoid(x), None),
            'denormalization': aux_funcs.sigmoid_denormalization
        }
    }

    def __init__(self, kernel_size, in_channels, stride=None, padding=0, dilation=1, ceil_mode=False):  #, normalization=None, denormalize=True):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        
        self.attention_conv = torch.nn.Conv2d(in_channels, 1, kernel_size=(1, 1), stride=1, padding=0)
        
        # if normalization not in self.available_normalizations.keys():
        #     raise Exception('Normalization {} unavailable for GroupingPool2d. Must be one of {}'.format(
        #         normalization, self.available_normalizations.keys()))
        # self.normalization = self.available_normalizations[normalization]['normalization']
        # self.denormalize = denormalize
        # self.denormalization = self.available_normalizations[normalization]['denormalization']

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)

        # Compute attention weights for all positions of the window:
        attention_weights = self.attention_conv(tensor)
        attention_weights = attention_weights.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        attention_weights = attention_weights.reshape((attention_weights.shape[0], attention_weights.shape[1], attention_weights.shape[2], attention_weights.shape[3],
                                                       self.kernel_size[0] * self.kernel_size[1]))
        
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))

                                 
        # # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        # output_tensor, normalization_params = self.normalization(tensor)
        # 4.-Compute reduction based as a weighted mean (using attention weights):
        output_tensor = (attention_weights * tensor).sum(dim=-1)

        # 5.-Denormalize output values after applying grouping
        # if self.denormalize:
        #     output_tensor = self.denormalization(output_tensor, normalization_params)

        return output_tensor


def pickPoolLayer(pool_option, initial_pool_exp=None):
    
    defaultOverlap2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, overlap=None, normalization='min_max', denormalize=True: OverlapPool2d(kernel_size, stride, padding, dilation, ceil_mode, overlap, normalization, denormalize)
    defaultGrouping2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True: GroupingPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize)
    defaultGroupingPlus2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True, initial_pool_exp=None: GroupingPlusPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize, initial_pool_exp=initial_pool_exp)
    defaultGroupingComposition2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_big=None, grouping_list=None, normalization='min_max', denormalize=True: GroupingCompositionPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping_big, grouping_list, normalization, denormalize)
    defaultGroupingComb2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_list=None, normalization='min_max', denormalize=True, learnable=True: GroupingCombPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping_list, normalization, denormalize, learnable)
    defaultUninorm2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, uninorm=None, normalization='min_max', denormalize=True: UninormPool2d(kernel_size, stride, padding, dilation, ceil_mode, uninorm, normalization, denormalize)
    defaultTnorm2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, tnorm=None, normalization='min_max', denormalize=True: TnormPool2d(kernel_size, stride, padding, dilation, ceil_mode, tnorm, normalization, denormalize)
    defaultTconorm2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, tconorm=None, normalization='min_max', denormalize=True: TconormPool2d(kernel_size, stride, padding, dilation, ceil_mode, tconorm, normalization, denormalize)
    defaultMD2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, deviation=None, normalization='linear', denormalize=True: MDPool2d(kernel_size, stride, padding, dilation, ceil_mode, deviation, normalization, denormalize)

    available_options = {
        'max': torch.nn.MaxPool2d,
        'avg': torch.nn.AvgPool2d,

        ### GROUPINGS:

        # DEBUGGING QUANTILE_NORMALIZATION:
        'grouping_product': lambda kernel_size, stride=None, padding=0, grouping='product': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),# , normalization='quantile'),
        'grouping_maximum': lambda kernel_size, stride=None, padding=0, grouping='maximum': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_minimum': lambda kernel_size, stride=None, padding=0, grouping='minimum': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_ob': lambda kernel_size, stride=None, padding=0, grouping='ob': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_u': lambda kernel_size, stride=None, padding=0, grouping='u': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        
        'grouping_plus_max': lambda kernel_size, stride=None, padding=0, grouping='max_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_plus_product': lambda kernel_size, stride=None, padding=0, grouping='product_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_plus_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),

        'grouping_comp_max_prodAndOB': lambda kernel_size, stride=None, padding=0, grouping_big='maximum', grouping_list=['product', 'ob']: defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list),
        'grouping_comp_prod_maxAndProd': lambda kernel_size, stride=None, padding=0, grouping_big='product', grouping_list=['maximum', 'product']: defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list),
        'grouping_comp_prod_maxAndOB': lambda kernel_size, stride=None, padding=0, grouping_big='product', grouping_list=['maximum', 'ob']: defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list),
        'grouping_comp_prod_prodAndOB': lambda kernel_size, stride=None, padding=0, grouping_big='product', grouping_list=['product', 'ob']: defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list),
        'grouping_comp_ob_maxAndProd': lambda kernel_size, stride=None, padding=0, grouping_big='ob', grouping_list=['maximum', 'product']: defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list),

        'grouping_comb_prodAndOB': lambda kernel_size, stride=None, padding=0, grouping_list=['product', 'ob']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_maxAndProd': lambda kernel_size, stride=None, padding=0, grouping_list=['maximum', 'product']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_maxAndOB': lambda kernel_size, stride=None, padding=0, grouping_list=['maximum', 'ob']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_maxProdAndOB': lambda kernel_size, stride=None, padding=0, grouping_list=['maximum', 'product', 'ob']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_maxAndGeometric': lambda kernel_size, stride=None, padding=0, grouping_list=['maximum', 'geometric']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),

        'grouping_comb_avgAndProd': lambda kernel_size, stride=None, padding=0, grouping_list=['average', 'product']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_avgAndOB': lambda kernel_size, stride=None, padding=0, grouping_list=['average', 'ob']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_avgAndGeometric': lambda kernel_size, stride=None, padding=0, grouping_list=['average', 'geometric']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        'grouping_comb_avgAndMax': lambda kernel_size, stride=None, padding=0, grouping_list=['average', 'maximum']: defaultGroupingComb2d(kernel_size, stride, padding, grouping_list=grouping_list),
        
        ### RESIDUAL GROUPINGS:
        'residual_group_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric': ResidualPool2d(kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_plus_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric_power': ResidualPool2d(kernel_size=kernel_size, stride=stride, pool_layer=defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_comp_prod_maxAndProd': lambda kernel_size, stride=None, padding=0, grouping_big='product', grouping_list=['maximum', 'product']: ResidualPool2d(kernel_size=kernel_size, stride=stride, pool_layer=defaultGroupingComposition2d(kernel_size, stride, padding, grouping_big=grouping_big, grouping_list=grouping_list)),
        # Testing different strategies for residual connection:
        'residual_group_prod_identity': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='identity', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_prod_mean': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='mean', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_prod_same_convex': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='same_convex', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_prod_diff_convex': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='diff_convex', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_prod_same': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='same', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),
        'residual_group_prod_diff': lambda kernel_size, stride=None, padding=0, grouping='product': ResidualPool2d(res_type='diff', kernel_size=kernel_size, stride=stride, pool_layer=defaultGrouping2d(kernel_size, stride, padding, grouping=grouping)),

        ### OVERLAPS:

        'overlap_product': lambda kernel_size, stride=None, padding=0, overlap='product': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap),# , normalization='quantile'),
        'overlap_minimum': lambda kernel_size, stride=None, padding=0, overlap='minimum': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap),
        'overlap_ob': lambda kernel_size, stride=None, padding=0, overlap='ob': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap),
        'overlap_geometric': lambda kernel_size, stride=None, padding=0, overlap='geometric': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap),
        'overlap_derivative': lambda kernel_size, stride=None, padding=0, overlap='product_derivative1_k': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap),

        ### OVERLAPS SIGMOID:

        'overlap_product_sigmoid': lambda kernel_size, stride=None, padding=0, overlap='product', normalization='sigmoid': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap, normalization=normalization),# , normalization='quantile'),
        'overlap_minimum_sigmoid': lambda kernel_size, stride=None, padding=0, overlap='minimum', normalization='sigmoid': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap, normalization=normalization),
        'overlap_ob_sigmoid': lambda kernel_size, stride=None, padding=0, overlap='ob', normalization='sigmoid': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap, normalization=normalization),
        'overlap_geometric_sigmoid': lambda kernel_size, stride=None, padding=0, overlap='geometric', normalization='sigmoid': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap, normalization=normalization),
        'overlap_derivative_sigmoid': lambda kernel_size, stride=None, padding=0, overlap='product_derivative1_k', normalization='sigmoid': defaultOverlap2d(kernel_size, stride, padding, overlap=overlap, normalization=normalization),

        'grouping_maximum_sigmoid': lambda kernel_size, stride=None, padding=0, grouping='maximum', normalization='sigmoid': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping, normalization=normalization),

        ### UNINORMS:
        'uninorm_min_max': lambda kernel_size, stride=None, padding=0, uninorm='min_max': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_product': lambda kernel_size, stride=None, padding=0, uninorm='product': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_lukasiewicz': lambda kernel_size, stride=None, padding=0, uninorm='lukasiewicz': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),

        ### T-NORMS:
        'tnorm_lukasiewicz': lambda kernel_size, stride=None, padding=0, tnorm='lukasiewicz': defaultTnorm2d(kernel_size, stride, padding, tnorm=tnorm),
        'tnorm_hamacher': lambda kernel_size, stride=None, padding=0, tnorm='hamacher': defaultTnorm2d(kernel_size, stride, padding, tnorm=tnorm),

        ### T-CONORMS
        'tconorm_maximum': lambda kernel_size, stride=None, padding=0, tconorm='maximum': defaultTconorm2d(kernel_size, stride, padding, tconorm=tconorm),
        'tconorm_prob_sum': lambda kernel_size, stride=None, padding=0, tconorm='prob_sum': defaultTconorm2d(kernel_size, stride, padding, tconorm=tconorm),
        'tconorm_bounded_sum': lambda kernel_size, stride=None, padding=0, tconorm='bounded_sum': defaultTconorm2d(kernel_size, stride, padding, tconorm=tconorm),
        'tconorm_hamacher': lambda kernel_size, stride=None, padding=0, tconorm='hamacher': defaultTconorm2d(kernel_size, stride, padding, tconorm=tconorm),
        'tconorm_einstein_sum': lambda kernel_size, stride=None, padding=0, tconorm='einstein_sum': defaultTconorm2d(kernel_size, stride, padding, tconorm=tconorm),

        ### MODERATE-DEVIATIONS
        'moderate_deviation': lambda kernel_size, stride=None, padding=0, deviation='test': defaultMD2d(kernel_size, stride, padding, deviation=deviation),
        'moderate_deviation_1_5': lambda kernel_size, stride=None, padding=0, deviation='m1_5': defaultMD2d(kernel_size, stride, padding, deviation=deviation),
        'moderate_deviation_2_5': lambda kernel_size, stride=None, padding=0, deviation='m2_5': defaultMD2d(kernel_size, stride, padding, deviation=deviation),
        'moderate_deviation_3': lambda kernel_size, stride=None, padding=0, deviation='m3': defaultMD2d(kernel_size, stride, padding, deviation=deviation),
        
        
        # DEBUG:
        ### ATTENTION-POOLING:
        'attention': AttentionPool2d,
        
    }

    return available_options[pool_option]


if __name__ == '__main__':
    print('Prueba')
    pruebaLayer = pickPoolLayer('grouping_product')(3, 1, 0)
    print(pruebaLayer)
