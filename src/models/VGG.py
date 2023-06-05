from functools import partial
from typing import Union, List, Dict, Any, Optional, cast

import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg16",
    "vgg16_bn",
]


class VGG_small(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x) # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x) # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, pool_layer: nn.Module = None, in_channels=3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            if pool_layer is None:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [pool_layer(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, pool_layer: nn.Module = None, small=False, in_channels=3, **kwargs: Any) -> VGG:
    if small:
        model = VGG_small(make_layers(cfgs[cfg], batch_norm=batch_norm, pool_layer=pool_layer, in_channels=in_channels), **kwargs)
    else:
        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, pool_layer=pool_layer, in_channels=in_channels), **kwargs)
    return model


def vgg16(pool_layer: nn.Module = None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        weights (VGG16_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("D", batch_norm=False, pool_layer=pool_layer, **kwargs)


def vgg16_bn(pool_layer: nn.Module = None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        weights (VGG16_BN_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("D", batch_norm=True, pool_layer=pool_layer, **kwargs)


def vgg16_small(pool_layer: nn.Module = None, **kwargs: Any) -> VGG_small:
    return _vgg("D", batch_norm=False, pool_layer=pool_layer, small=True, **kwargs)

def vgg16_bn_small(pool_layer: nn.Module = None, **kwargs: Any) -> VGG_small:
    return _vgg("D", batch_norm=True, pool_layer=pool_layer, small=True, **kwargs)
