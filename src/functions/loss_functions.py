import torch
import torch.nn as nn

class SupervisedCrossEntropyLoss(nn.Module):

    def __init__(self, num_epochs, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', alpha=1):
        super().__init__()
        self.crossEntropyLoss = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)
        self.num_epochs = num_epochs
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, epoch):
        input_final, inputs_supervised = inputs
        loss = self.crossEntropyLoss(input_final, target)
        for input_supervised in inputs_supervised:
            loss += self.crossEntropyLoss(input_supervised) * (self.alpha * 0.1 * (1 - float(epoch / self.num_epochs)))
        return loss.contiguous()