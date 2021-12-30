import torch.nn as nn

# Debug:
import torch

from math import ceil, floor
import src.layers.pooling_layers as pool_layers


# ToDO: Refactorize this layers to a separate module:
#  Auxiliar layers:

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape([input.size(0), -1])


class HiddenLayerSupervision(nn.Module):
    def __init__(self, in_features, num_classes, classifier_name='softmax'):
        super(HiddenLayerSupervision, self).__init__()
        if classifier_name == 'softmax':
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(in_features, num_classes, bias=True)
            )
        else:
            # ToDo: Complete
            pass

    def forward(self, input):
        return self.classifier(input)


class SupervisedNiNPlus(nn.Module):

    activation_functions = {
        'relu': nn.ReLU,
        'leaky': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'mish': nn.Mish,
    }

    network_params = {
        'conv_filters': (128, 128, 192, 192, 256, 256),
        'mlpconv_neurons': (128, 192, 256)
    }

    def __init__(self, pool_layer=nn.MaxPool2d, supervision_type='softmax', in_channels=3, num_classes=10, input_size=(32, 32), initial_pool_exp=None, activation='relu'):
        '''Constructor method

        :param pool_functions: pool functions to be used for the pooling phase. If None, MaxPool2d will be used.
        :type pool_functions: iterable, optional
        :param dataset: Specifies the dataset to be used for training the model, defaults to 'CIFAR10'
        :type str, optional
        :param supervision_type: Specifies classifier to be used for hidden layer supervision, defaults to 'softmax'
        :type str, optional
        '''
        super().__init__()

        # Block 1:
        # ToDo: Debug the following padding calculus
        block_1_pool_pad = (
            floor((input_size[0] - 3) % 2 / 2.0), ceil((input_size[0] - 3) % 2 / 2.0),
            floor((input_size[1] - 3) % 2 / 2.0), ceil((input_size[1] - 3) % 2 / 2.0))
        # Block 1:
        #self.relu = nn.ReLU(inplace=True);
        self.block_1_conv1 = nn.Conv2d(in_channels, self.network_params['conv_filters'][0], kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.ReLU(inplace=True)
        self.block_1_supervision1 = HiddenLayerSupervision(
            input_size[0] * input_size[1] * self.network_params['conv_filters'][0], num_classes, classifier_name=supervision_type
        )
        self.block_1_conv2 = nn.Conv2d(self.network_params['conv_filters'][0], self.network_params['conv_filters'][1],
                                       kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU(inplace=True)
        self.block_1_supervision2 = HiddenLayerSupervision(
            input_size[0] * input_size[1] * self.network_params['conv_filters'][1], num_classes, classifier_name=supervision_type
        )
        self.block_1_mlpconv = nn.Conv2d(self.network_params['conv_filters'][1], self.network_params['mlpconv_neurons'][0],
                                         kernel_size=1, stride=1, bias=True)
        # self.relu3 = nn.ReLU(inplace=True)

        if pool_layer in (nn.MaxPool2d, nn.AvgPool2d):
            self.block_1_pool = pool_layer(kernel_size=3, stride=2, ceil_mode=True)
        # Empirical tests show that the initial value of the parameter p of GroupingPlusPool2d layers is irrelevant
        # elif pool_layer == pool_layers.GroupingPlusPool2d:
        #     self.block_1_pool = pool_layer(kernel_size=3, stride=2, padding=block_1_pool_pad, initial_pool_exp=initial_pool_exp)
        else:
            self.block_1_pool = pool_layer(kernel_size=3, stride=2, padding=block_1_pool_pad)

        self.block_1_dropout = nn.Dropout2d(p=0.5, inplace=True)

        block_2_pool_pad = (
            floor((input_size[0]//2 - 3) % 2 / 2.0), ceil((input_size[0]//2 - 3) % 2 / 2.0),
            floor((input_size[1]//2 - 3) % 2 / 2.0), ceil((input_size[1]//2 - 3) % 2 / 2.0))
        # Block 2:
        self.block_2_conv1 = nn.Conv2d(self.network_params['mlpconv_neurons'][0], self.network_params['conv_filters'][2],
                                       kernel_size=3, stride=1, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        self.block_2_supervision1 = HiddenLayerSupervision(
            (input_size[0]//2) * (input_size[1]//2) * self.network_params['conv_filters'][2], num_classes,
            classifier_name=supervision_type
        )
        self.block_2_conv2 = nn.Conv2d(self.network_params['conv_filters'][2], self.network_params['conv_filters'][3],
                                       kernel_size=3, stride=1, padding=1)
        # self.relu5 = nn.ReLU(inplace=True)
        self.block_2_supervision2 = HiddenLayerSupervision(
            (input_size[0]//2) * (input_size[1]//2) * self.network_params['conv_filters'][3], num_classes,
            classifier_name=supervision_type
        )
        self.block_2_mlpconv = nn.Conv2d(self.network_params['conv_filters'][3], self.network_params['mlpconv_neurons'][1],
                                         kernel_size=1, stride=1, bias=True)
        # self.relu6 = nn.ReLU(inplace=True)
        if pool_layer in (nn.MaxPool2d, nn.AvgPool2d):
            self.block_2_pool = pool_layer(kernel_size=3, stride=2, ceil_mode=True)
        # Empirical tests show that the initial value of the parameter p of GroupingPlusPool2d layers is irrelevant
        # elif pool_layer == pool_layers.GroupingPlusPool2d:
        #     self.block_2_pool = pool_layer(kernel_size=3, stride=2, padding=block_2_pool_pad, initial_pool_exp=initial_pool_exp)
        else:
            self.block_2_pool = pool_layer(kernel_size=3, stride=2, padding=block_2_pool_pad)
        self.block_2_dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.block_3_conv1 = nn.Conv2d(self.network_params['mlpconv_neurons'][1], self.network_params['conv_filters'][4],
                                       kernel_size=3, stride=1, padding=1)
        # self.relu7 = nn.ReLU(inplace=True)
        self.block_3_supervision1 = HiddenLayerSupervision(
            (input_size[0]//4) * (input_size[1]//4) * self.network_params['conv_filters'][4], num_classes,
            classifier_name=supervision_type
        )
        self.block_3_conv2 = nn.Conv2d(self.network_params['conv_filters'][4], self.network_params['conv_filters'][5],
                                       kernel_size=3, stride=1, padding=1)
        # self.relu8 = nn.ReLU(inplace=True)
        self.block_3_supervision2 = HiddenLayerSupervision(
            (input_size[0]//4) * (input_size[1]//4) * self.network_params['conv_filters'][5], num_classes,
            classifier_name=supervision_type
        )
        self.block_3_mlpconv1 = nn.Conv2d(self.network_params['conv_filters'][5], self.network_params['mlpconv_neurons'][2],
                                          kernel_size=1, stride=1, bias=True)
        self.block_3_mlpconv2 = nn.Conv2d(self.network_params['mlpconv_neurons'][2], num_classes, kernel_size=1,
                                          stride=1, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  #AdaptiveAvgPool2d can be used as Global Average Pooling
            # if output_size is set to 1. This will make sure to compute the average of all values by channel and avoids
            # having to set the size of the output up at to this point (which may vary depending on the used dataset).

        try:
            self.activation = self.activation_functions[activation](inplace=True)
        except TypeError:
            self.activation = self.activation_functions[activation]()


    def forward(self, input):
        if self.training:
            x_supervised = []
        # Block 1:
        x = self.block_1_conv1(input)
        # x = self.relu1(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_1_supervision1(x))
        x = self.block_1_conv2(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_1_supervision2(x))
        x = self.block_1_mlpconv(x)
        x = self.activation(x)
        x = self.block_1_pool(x)
        x = self.block_1_dropout(x).contiguous()
        # Block 2:
        x = self.block_2_conv1(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_2_supervision1(x))
        x = self.block_2_conv2(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_2_supervision2(x))
        x = self.block_2_mlpconv(x)
        x = self.activation(x)
        x = self.block_2_pool(x)
        x = self.block_2_dropout(x).contiguous()
        # Block 3:
        x = self.block_3_conv1(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_3_supervision1(x))
        x = self.block_3_conv2(x)
        x = self.activation(x)
        if self.training:
            x_supervised.append(self.block_3_supervision2(x))
        x = self.block_3_mlpconv1(x)
        x = self.block_3_mlpconv2(x)
        x = self.avg_pool(x)

        x = x.reshape([x.shape[0], x.shape[1]])

        # # Debug:
        # some_nan = torch.sum(torch.isnan(x)) > 0
        # if self.training:
        #     for x_superv in x_supervised:
        #         if torch.sum(torch.isnan(x_superv)) > 0:
        #             some_nan = True
        # if some_nan:
        #     print('Error')

        if self.training:
            return (x, x_supervised)
        else:
            return x