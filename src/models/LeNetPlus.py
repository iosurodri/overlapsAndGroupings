import torch.nn as nn


class LeNetPlus(nn.Module):

    def __init__(self, input_size, output_size, num_filters=[64, 64],
                 fc_sizes=[384, 192, -1], pool_layer=nn.MaxPool2d, use_batch_norm=True):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.use_batch_norm = use_batch_norm

        # Layer "group" 0:
        self.conv_0 = nn.Conv2d(in_channels=input_size[2], out_channels=num_filters[0],
                                   kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.pool_0 = pool_layer(kernel_size=(2, 2), stride=(2, 2))
        if use_batch_norm:
            self.batch_norm_0 = nn.BatchNorm2d(num_filters[0])
        # Layer group 1:
        self.conv_1 = nn.Conv2d(num_filters[0], out_channels=num_filters[1],
                                kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.pool_1 = pool_layer(kernel_size=(2, 2), stride=(2, 2))
        if use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(num_filters[1])
        # MLP Classifier:
        self.flatten = nn.Flatten()
        flattened_size = int((input_size[0] / (2 * 2)) * (input_size[1] / (2 * 2))) * num_filters[1]
        self.fc_0 = nn.Linear(flattened_size, fc_sizes[0], bias=True)
        self.fc_1 = nn.Linear(fc_sizes[0], fc_sizes[1], bias=True)
        self.fc_2 = nn.Linear(fc_sizes[1], output_size, bias=True)

    def forward(self, x):
        output_block_0 = self.pool_0(self.relu(self.conv_0(x)))
        if self.use_batch_norm:
             output_block_0 = self.batch_norm_0(output_block_0)
        output_block_1 = self.pool_1(self.relu(self.conv_1(output_block_0)))
        if self.use_batch_norm:
            output_block_1 = self.batch_norm_1(output_block_1)
        output_flattened = self.flatten(output_block_1)
        output_fc_0 = self.relu(self.fc_0(output_flattened))
        output_fc_1 = self.relu(self.fc_1(output_fc_0))
        predictions = self.fc_2(output_fc_1)  # Softmax will be applied to this value
        return predictions