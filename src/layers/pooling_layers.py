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
        'product': torch.prod,
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
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

        if overlap not in self.available_groupings.keys():
            raise Exception('Overlap {} unavailable for OverlapPool2d. Must be one of {}'.format(
                overlap, self.available_groupings.keys()))
        self.grouping = self.available_groupings[overlap]
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
        tensor = tensor.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size * self.kernel_size))
        # 3.-ToDo: Normalize the input so that overlaps defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # output_tensor = tensor
        # 4.-Compute reduction based on the chosen overlap:
        output_tensor = self.overlap(output_tensor, dim=-1)
        # output_tensor = torch.max(output_tensor, dim=-1)[0]

        # ToDo: Denormalize output values after applying overlap?
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
        stride = stride if (stride is not None) else kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride
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
                 num_channels=1):
        super().__init__()
        self.kernel_size = kernel_size
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
            self.weight = torch.nn.Parameter(torch.ones([1, 1, 1, 1, 1]) * 0.5)
        elif weight_mode == 'channel_wise':
            self.weight = torch.nn.Parameter(torch.ones([1, num_channels, 1, 1, 1]) * 0.5)
        else:
            raise Exception('Wrong option for weight_mode provided: {}'.format(weight_mode))
        

    def forward(self, tensor):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        tensor = tensor.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # 2.-Turn each one of those 2D patches into a 1D vector:
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size * self.kernel_size))
        # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
        output_tensor, normalization_params = self.normalization(tensor)
        # If automorphisms are to be used, apply them before applying the grouping (if chosen to do so):
        
        # 4.-Compute reduction based on the chosen grouping:
        output_tensor = self.grouping(output_tensor, dim=-1)

        # 5.-Denormalize output values after applying grouping
        if self.denormalize:
            output_tensor = self.denormalization(output_tensor, normalization_params)
        
        if self.transform_before and self.automorphism_weight is not None:
            output_tensor = torch.pow(output_tensor+0.000001, self.automorphism_weight) 
        # 4.-Compute reduction based on the chosen grouping:
        output_tensor = self.grouping(output_tensor, dim=-1)
        # If automorphisms are to be used, apply them after applying the grouping (if chosen to do so):
        if not self.transform_before and self.automorphism_weight is not None:
            output_tensor = torch.pow(output_tensor+0.000001, self.automorphism_weight[:, 0])
        

        return output_tensor



# class GroupingPlusPool2d(torch.nn.Module):

#     available_groupings = {
#         'product': aggr_funcs.product_grouping,
#         'maximum': lambda x: torch.max(x)[0],
#         'minimum': aggr_funcs.minimum_grouping,
#         'ob': aggr_funcs.ob_grouping,
#         'geometric': aggr_funcs.geometric_grouping
#     }

#     available_normalizations = {
#         'min_max': {
#             'normalization': aux_funcs.min_max_normalization,
#             'denormalization': aux_funcs.min_max_denormalization
#         }
#     }

#     def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization=None, denormalize=True, automorphism_type='same', transform_before=True):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride if (stride is not None) else kernel_size
#         self.padding = padding
#         self.dilation = dilation
#         self.ceil_mode = ceil_mode
#         if grouping not in self.available_groupings.keys():
#             raise Exception('Grouping {} unavailable for GroupingPlusPool2d. Must be one of {}'.format(
#                 grouping, self.available_groupings.keys()))
#         self.grouping = self.available_groupings[grouping]
#         if normalization not in self.available_normalizations.keys():
#             raise Exception('Normalization {} unavailable for GroupingPlusPool2d. Must be one of {}'.format(
#                 normalization, self.available_normalizations.keys()))
#         self.normalization = self.available_normalizations[normalization]['normalization']
#         self.denormalize = denormalize
#         self.denormalization = self.available_normalizations[normalization]['denormalization']
#         # Parameters which model the automorphishms to apply to the 
#         if automorphism_type == 'same':
#             # self.automorphism_weight = torch.nn.Parameter(torch.rand(1, 1, 1, 1, 1))
#             self.automorphism_weight = torch.nn.Parameter(torch.ones([1, 1, 1, 1, 1]) * 0.5)
#         elif automorphism_type == 'distinct':
#             self.automorphism_weight = torch.nn.parameter(torch.rand(1, 1, 1, 1, kernel_size*kernel_size))
#         else:
#             raise Exception('Unavailable automorphism_type. Must be either "same" or "disctinct"')
#         self.transform_before = transform_before
        

#     def forward(self, tensor):
#         if isinstance(self.padding, list) or isinstance(self.padding, tuple):
#             tensor = F.pad(tensor, self.padding)
#         # 1.-Extract patches of kernel_size from tensor:
#         tensor = tensor.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         # 2.-Turn each one of those 2D patches into a 1D vector:
#         tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
#                                  self.kernel_size * self.kernel_size))
#         # 3.-ToDo: Normalize the input so that groupings defined in (0, 1) can be properly applied:
#         output_tensor, normalization_params = self.normalization(tensor)
#         # If automorphisms are to be used, apply them before applying the grouping (if chosen to do so):
#         if self.transform_before and self.automorphism_weight is not None:
#             output_tensor = torch.pow(output_tensor+0.000001, self.automorphism_weight) 
#         # 4.-Compute reduction based on the chosen grouping:
#         output_tensor = self.grouping(output_tensor, dim=-1)
#         # If automorphisms are to be used, apply them after applying the grouping (if chosen to do so):
#         if not self.transform_before and self.automorphism_weight is not None:
#             output_tensor = torch.pow(output_tensor+0.000001, self.automorphism_weight[:, 0])
#         # 5.-Denormalize output values after applying grouping
#         if self.denormalize:
#             output_tensor = self.denormalization(output_tensor, normalization_params)

#         return output_tensor


def pickPoolLayer(pool_option):
    
    defaultOverlap2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, overlap=None, normalization='min_max', denormalize=True: OverlapPool2d(kernel_size, stride, padding, dilation, ceil_mode, overlap, normalization, denormalize)
    defaultGrouping2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True: GroupingPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize)
    defaultGroupingPlus2d = lambda kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, grouping=None, normalization='min_max', denormalize=True: GroupingPlusPool2d(kernel_size, stride, padding, dilation, ceil_mode, grouping, normalization, denormalize)

    available_options = {
        'max': torch.nn.MaxPool2d,
        'avg': torch.nn.AvgPool2d,
        'grouping_product': lambda kernel_size, stride=None, padding=0, grouping='product': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_maximum': lambda kernel_size, stride=None, padding=0, grouping='maximum': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_minimum': lambda kernel_size, stride=None, padding=0, grouping='minimum': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_ob': lambda kernel_size, stride=None, padding=0, grouping='ob': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_u': lambda kernel_size, stride=None, padding=0, grouping='u': defaultGrouping2d(kernel_size, stride, padding, grouping=grouping),
        
        'grouping_plus_max': lambda kernel_size, stride=None, padding=0, grouping='max_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_plus_product': lambda kernel_size, stride=None, padding=0, grouping='product_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),
        'grouping_plus_geometric': lambda kernel_size, stride=None, padding=0, grouping='geometric_power': defaultGroupingPlus2d(kernel_size, stride, padding, grouping=grouping),
    }

    return available_options[pool_option]


if __name__ == '__main__':
    print('Prueba')
    pruebaLayer = pickPoolLayer('grouping_product')(3, 1, 0)
    print(pruebaLayer)
