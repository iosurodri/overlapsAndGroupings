import os
import sys
sys.path.append(os.path.join(os.path.realpath('.'), os.pardir, os.pardir))

import torch
import torch.nn.functional as F

import os
print(os.curdir)

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

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization=None, denormalize=True):
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

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
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
        if initial_pool_exp is None:
            initial_pool_exp = 0.25
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
        output_tensor = self.grouping(output_tensor, self.weight, dim=-1)

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
        'minimum': lambda x, dim=-1, keepdim=False: torch.min(x, keepdim=keepdim, dim=dim),
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


class UninormPool2d(torch.nn.Module):

    available_uninorms = {
        'product': lambda x, dim=-1, keepdim=False: torch.prod(x, keepdim=keepdim, dim=dim),
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
        self.uinorm = self.available_uninorms[uninorm]
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



def pickPoolLayer(pool_option, initial_pool_exp=None):
    
    defaultOverlap2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, overlap=None, normalization='min_max', denormalize=True: OverlapPool2d(kernel_size, stride, padding, dilation, ceil_mode, overlap, normalization, denormalize)
    defaultGrouping2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True: GroupingPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize)
    defaultGroupingPlus2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True, initial_pool_exp=None: GroupingPlusPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize, initial_pool_exp=initial_pool_exp)
    defaultGroupingComposition2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_big=None, grouping_list=None, normalization='min_max', denormalize=True: GroupingCompositionPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping_big, grouping_list, normalization, denormalize)
    defaultGroupingComb2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping_list=None, normalization='min_max', denormalize=True, learnable=False: GroupingCombPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping_list, normalization, denormalize, learnable)
    defaultUninorm2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, uninorm=None, normalization='min_max', denormalize=True: UninormPool2d(kernel_size, stride, padding, dilation, ceil_mode, uninorm, normalization, denormalize)
    defaultTnorm2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, tnorm=None, normalization='min_max', denormalize=True: TnormPool2d(kernel_size, stride, padding, dilation, ceil_mode, tnorm, normalization, denormalize)

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
        # DEBUG: Important -> Debugging the influence of the initial exponent for the pooling layer
        'grouping_plus_product': lambda kernel_size, stride=None, padding=0, grouping='product_power', initial_pool_exp=initial_pool_exp: defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping, initial_pool_exp=initial_pool_exp),
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
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),
        'uninorm_XXX': lambda kernel_size, stride=None, padding=0, uninorm='XXX': defaultUninorm2d(kernel_size, stride, padding, uninorm=uninorm),

        ### TNORMS:
        'tnorm_lukasiewicz': lambda kernel_size, stride=None, padding=0, tnorm='lukasiewicz': defaultTnorm2d(kernel_size, stride, padding, tnorm=tnorm),
        'tnorm_hamacher': lambda kernel_size, stride=None, padding=0, tnorm='hamacher': defaultTnorm2d(kernel_size, stride, padding, tnorm=tnorm)
    }

    return available_options[pool_option]


if __name__ == '__main__':
    print('Prueba')
    pruebaLayer = pickPoolLayer('grouping_product')(3, 1, 0)
    print(pruebaLayer)
