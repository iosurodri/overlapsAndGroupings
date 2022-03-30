from multiprocessing import Pool, pool
import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
### AUXILIAR FUNCTIONS ###
##########################

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


###############################
### BASELINE IMPLEMENTATION ###
###############################

class BasicBlockSmall(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bottleneck_option='pad_constant'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if bottleneck_option == 'pad_constant':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # Add padding in the "depth" dimension to deal with increase in the number of planes (from input to output). 
                # x[:, :, ::2, ::2] ensures that the downsampled values will be ommited (similarly to stride=2).
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                # NOTE: planes//4 only works because we assume an exponential increase in the number of filters (as a power of 2)
                # A more general implementation could use (planes-in_planes) // 2 as an alternative valid for any increase on the number of parameters.
            elif bottleneck_option == 'conv_bottleneck':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetSmall(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bottleneck_option='pad_constant'):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, bottleneck_option=bottleneck_option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, bottleneck_option=bottleneck_option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, bottleneck_option=bottleneck_option)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, bottleneck_option='pad_constant'):
        strides = [stride] + [1]*(num_blocks-1)  # A stride of 1 is used for all convolutions but (potentially) the first one
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bottleneck_option=bottleneck_option))  # Number of channels is always in_planes (will be updated in next code line if necessary)
            self.in_planes = planes * block.expansion  # self.in_planes is updated every time the number of channels of the input increases

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

##############################
### POOLING IMPLEMENTATION ###
##############################

class PoolBlockSmall(nn.Module):
    expansion = 1

    bottleneck_options = {
        # If there is an increase in the number of channels from the input to the output of the block,
        # additional channels will be created, "padding" them with either:
        'pad_constant': lambda planes, in_planes: LambdaLayer(lambda x:
            F.pad(x, (0, 0, 0, 0, (planes-in_planes)//2, (planes-in_planes)//2), "constant", 0)),
        # F.pad(mode="repcicate") available only for padding left, right, top and bottom
        # An alternative that replicates the intended behaviour would be:
        'pad_replicate': lambda planes, in_planes: LambdaLayer(lambda x:
            torch.repeat_interleave(x, planes // in_planes, dim=1)),
        'conv_bottleneck': lambda planes, in_planes: nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )
    }


    def __init__(self, in_planes, planes, bottleneck_option='pad_constant'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if in_planes != planes:
            if bottleneck_option not in self.bottleneck_options.keys():
                raise Exception('Unavailable bottleneck option {} requested'.format(bottleneck_option))
            self.shortcut = self.bottleneck_options[bottleneck_option](planes, in_planes)


    def forward(self, x):
        # ToDo: Add the pooling function as a parameter of the forward pass (allows for )
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetPool(nn.Module):

    # Configurations for number of filters:
    planes_configurations = {
        'double':      [16, 32, 64],  # Standard implementation (number of planes doubles after each downsampling)
        'constant_16': [16, 16, 16],
        'constant_32': [32, 32, 32]
    } 

    def __init__(self, block, num_blocks, num_classes=10, planes_conf='double', pool_layer=nn.MaxPool2d, bottleneck_option='pad_constant'):

        super().__init__()
        
        if planes_conf not in self.planes_configurations.keys():
            raise Exception('TODO') # TODO
        planes_configuration = self.planes_configurations[planes_conf]

        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, planes_configuration[0], num_blocks[0], bottleneck_option=bottleneck_option)
        # First downsampling:
        self.pool1 = pool_layer(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(block, planes_configuration[1], num_blocks[1], bottleneck_option=bottleneck_option)
        # Second downsampling:
        self.pool2 = pool_layer(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(block, planes_configuration[2], num_blocks[2], bottleneck_option=bottleneck_option)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, bottleneck_option = 'pad_constant'):
        # Downsampling is taken care of outside of the Sequential blocks (differently to the rest of implementations)
        layers = []
        for i in range(num_blocks):
            # NOTE: Since downsampling is not performed through convolution, stride is always 1
            layers.append(block(self.in_planes, planes, bottleneck_option=bottleneck_option))  # Number of channels is always in_planes (will be updated in next code line if necessary)
            self.in_planes = planes * block.expansion  # self.in_planes is updated every time the number channels of the input increases
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # First downsampling:
        out = self.pool1(out)
        out = self.layer2(out)#, pool_fn=self.pool1)
        # Second downsampling:
        out = self.pool2(out)
        out = self.layer3(out)#, pool_fn=self.pool2)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

###########################
### MODEL INSTANTIATION ###
###########################

def baseline_resnet20_pad_bottleneck() -> ResNetSmall:
    return ResNetSmall(BasicBlockSmall, [3, 3, 3], bottleneck_option='pad_constant')

def baseline_resnet20_conv_bottleneck() -> ResNetSmall:
    return ResNetSmall(BasicBlockSmall, [3, 3, 3], bottleneck_option='conv_bottleneck')

def pool_resnet20(pool_layer, planes_conf='double', bottleneck_option='pad_constant') -> ResNetPool:
    return ResNetPool(PoolBlockSmall, [3, 3, 3], pool_layer=pool_layer, planes_conf=planes_conf, bottleneck_option=bottleneck_option)


def get_resnet(model_type, bottleneck_option, planes_conf, pool_layer=None, size=20):

    # available_models = {
    #     'small': {
    #         'model': ResNetSmall,
    #         'block': BasicBlockSmall
    #     },
    #     'pool': {
    #         'model': ResNetPool,
    #         'block': PoolBlockSmall
    #     }
    # }

    available_sizes = {
        20: [3, 3, 3],
        32: [5, 5, 5],
        44: [7, 7, 7],
        56: [9, 9, 9]
    }

    if model_type == 'small':
        return ResNetSmall(BasicBlockSmall, available_sizes[size], bottleneck_option=bottleneck_option)
    elif model_type == 'pool':
        return ResNetPool(PoolBlockSmall, available_sizes[size], planes_conf=planes_conf, bottleneck_option=bottleneck_option, pool_layer=pool_layer)
    else:
        raise Exception('Unavailable model_type {} provided'.format(model_type))

if __name__ == '__main__':

    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join('..', '..')))

    test_input = torch.rand([2, 3, 32, 32], dtype=torch.float, device='cuda')
    bottleneck_option = 'pad_constant'

    import src.layers.pooling_layers as pool_layers
    # pool_layer = nn.MaxPool2d
    pool_layer = pool_layers.pickPoolLayer('grouping_product')
    model_pool = baseline_resnet20_pad_bottleneck().to('cuda')
    # model_pool = pool_resnet20(pool_layer, bottleneck_option=bottleneck_option).to('cuda')
    
    output = model_pool(test_input)