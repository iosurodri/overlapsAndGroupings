import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from typing import Any, List

from src.visualization.visualize_distributions import visualize_heatmap, visualize_hist


# AUXILIARY LAYERS DEFINITION:

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        # Bottleneck for dimensionality reduction (conv 1x1) -> Sets next convolution number of input features to
        # bn_size * growth_rate
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        # Regular convolution operation
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        # Bottleneck for alleviating the work of the following convolution layer by performing dimensionality reduction:
        concated_features = torch.cat(inputs, 1)  # Stacks all input feature maps into a single input tensor
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    # Checks if any of the tensors in input requires grad computation:
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # FIX: Previously input was sent as a list to cp.checkpoint (cp.checkpoint(closure, input)), but CheckpointFunction
    # receives n args with the * operator and creates a list already, so a list of lists (instead of tensors) was being
    # analized. With the new change, the closure function has had to been changed to, in order for bn_function to
    # get a list as input parameter instead of a list.
    # Alternative for direct bn_function() call when memory constraints are a serious problem:
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    # Forward method of the layer:
    def forward(self, input):
        if isinstance(input, Tensor):
            # If this is one of the first layers that only receive an image or feature map (instead of all the already
            # computed feature maps):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            # We sacrifice computation time for memory (these values will be recomputed during backward)
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool_layer=nn.AvgPool2d):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', pool_layer(kernel_size=2, stride=2))


# DENSE BLOCK DEFINITION (Multiple DenseLayers):

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)  # Return a single tensor created by concatenating all features maps, in order for
        # common layers to be able to operate (without requiring special logic such as the one of _DenseLayer)


class DenseNetPlus(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    #def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
    #             num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
    def __init__(self, growth_rate=12, num_layers=100, bn_size=4, drop_rate=0.2, num_classes=10,
                 memory_efficient=False, pool_layer=nn.AvgPool2d, in_channels=3, classifier_layers=1):

        super().__init__()

        # Note: notice that the number of layers in each block is: ((num_layers - 4) / 3) / 2
        # 4 is the number of layers extra layers (1 in the start + 2 transition layers + 1 as output)
        # 3 is the number of blocks in the network
        # 2 is added because we use bottleneck, which means that bottleneck layers are half of all layers
        if num_layers == 40:
            block_config = (12, 12, 12)
        elif num_layers == 100:
            block_config = (32, 32, 32)
        elif num_layers == 190:
            block_config = (64, 64, 64)
        else:
            raise Exception('Please, set num_layers to one of the following values: 40, 100, 190')

        num_init_features = 2 * growth_rate  # Based in the original article, and since we use the version with both bottleneck and compression

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate  # Each layer has added growth_rate layers to the
            # global knowledge of the network (the total number of feature images produced by any layer)
            # We add a Transition Layer between all dense blocks (except at the end of the network):
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,  # Compression of 0.5
                                    pool_layer=pool_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2  # We have compressed the feature image by a factor of 0.5

        # Final batch norm
        self.features.add_module('norm4', nn.BatchNorm2d(num_features))

        # Linear layer
        if classifier_layers == 1:
            self.classifier = nn.Linear(num_features, num_classes)
        elif classifier_layers == 2:
            self.classifier = nn.Sequential(OrderedDict([
                ('linear0', nn.Linear(num_features, 100)),
                ('relu0', nn.ReLU(inplace=True)),
                ('linear1', nn.Linear(100, num_classes))
            ]))
        else:
            raise Exception('Unavailable number of classifier layers for this model: classifier_layers must be 1 or 2.')
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # In this case it acts as a common AvgPool2d, but doesn't require to
        # specify the kernel size.
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class BigDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, num_layers=121, num_init_features=None, bn_size=4, drop_rate=0, num_classes=3,
                 memory_efficient=False, pool_layer=nn.AvgPool2d, in_channels=3, classifier_layers=1):

        super(BigDenseNet, self).__init__()

        if num_layers == 121:
            block_config = (6, 12, 24, 16)
        elif num_layers == 169:
            block_config = (6, 12, 32, 32)
        elif num_layers == 201:
            block_config = (6, 12, 48, 32)
        elif num_layers == 264:
            block_config = (6, 12, 64, 48)
        else:
            raise Exception('Please, set num_layers to one of the following values: 121, 169, 201, 264')

        if num_init_features is None:
            num_init_features = 2 * growth_rate

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    pool_layer=pool_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier_layers = classifier_layers
        if classifier_layers == 1:
            self.classifier = nn.Linear(num_features, num_classes)
        elif classifier_layers == 2:
            # self.classifier = nn.Sequential(OrderedDict([
            #     ('classifier_linear0', nn.Linear(num_features, 100)),
            #     # ('classifier_norm', nn.BatchNorm2d(100)),
            #     ('classifier_relu0', nn.ReLU(inplace=True)),
            #     ('classifier_linear1', nn.Linear(100, num_classes))
            # ]))
            self.classifier_linear0 = nn.Linear(num_features, 256)
            self.classifier_bn = nn.BatchNorm1d(256)
            self.classifier_relu = nn.ReLU(inplace=True)
            self.classifier_linear1 = nn.Linear(256, num_classes)
        else:
            raise Exception('Unavailable number of classifier layers for this model: classifier_layers must be 1 or 2.')

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        #out = self.classifier(out)
        if self.classifier_layers == 1:
            out = self.self.classifier(out)
        elif self.classifier_layers == 2:
            out = self.classifier_linear0(out)
            out = self.classifier_bn(out)
            out = self.classifier_relu(out)
            out = self.classifier_linear1(out)
        else:
            raise Exception('Classifier must have 1 or 2 layers.')
        return out


# TRANSFER LEARNING DEFINITION:

def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            # Some parameters, as obtained from load_state_dict_from_url() are of the form:
            # features.denseblockI.denselayerJ.norm.K.weight. We require them in this other format:
            # features.denseblockI.denselayerJ.normK.weight in order for the model to be able to load them:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    # DEBUG: Original number of classes is 1000, while now we work with just 4 classes:
    del state_dict['classifier.weight']
    del state_dict['classifier.bias']
    model.load_state_dict(state_dict, strict=False)  # Debug: Added strict=False


def _densenet(
    arch: str,
    growth_rate: int,
    # block_config: Tuple[int, int, int, int],
    num_layers: int,
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    pool_layer=nn.AvgPool2d,
    **kwargs: Any
) -> DenseNetPlus:
    model = BigDenseNet(growth_rate, num_layers, num_init_features, pool_layer=pool_layer,**kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def load_pretrained_densenet(num_layers: int = 121, progress: bool = True, freeze_layers: bool = False, pool_layer=nn.AvgPool2d, **kwargs: Any) -> BigDenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        num_layers (int): Number of layers of the model. Only certain values are allowed: 121, 169 and 201
        progress (bool): If True, displays a progress bar of the download to stderr
        exclude_classifier (bool): If True, Parameters associated with the model's classifier won't be frozen.
        exclude_pool_coeff (bool): If True, coefficients associated with the PoolComb coefficients won't be frozen.
    """
    # NOTE: Growth rate for all the available pretrained models is 32
    available_models = (121, 169, 201, 264)
    if num_layers not in available_models:
        raise Exception('Number of layers must be one of: {}'.format(available_models))
    model = _densenet('densenet{}'.format(str(num_layers)), growth_rate=32, num_layers=num_layers, num_init_features=64,
                      pretrained=True, progress=progress, pool_layer=pool_layer, **kwargs)
    for param_name, parameter in model.named_parameters():
        # By default, freeze all model parameters:
        freeze_parameter = True
        if freeze_layers:
            # Do not freeze Parameters associated to the model's classifier:
            if 'classifier' in param_name:
                freeze_parameter = False
            # Do not freeze Parameters associated to PoolComb layers:
            if 'pool.weight' in param_name:
                freeze_parameter = False
            if freeze_parameter:
                parameter.requires_grad = False
    return model


if __name__ == '__main__':
    model = load_pretrained_densenet(num_layers=121, progress=True, drop_rate=0, num_classes=4,
                                     memory_efficient=False, pool_layer=nn.AvgPool2d,
                                     in_channels=3, exclude_classifier=True, exclude_pool_coeff=True)
    print(model)
