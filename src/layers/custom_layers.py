from turtle import forward
import torch
import torch.nn as nn


class AnomalyRemover(nn.Module):

    def __init__(self, channelwise=True):
        super().__init__()
        self.channelwise = channelwise

    def forward(self, in_tensor):
        if self.channelwise:
            # Can't replace the wrong values through a direct assignment, since that would break the backward pass:
            