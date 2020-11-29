import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseSoftmax(nn.Module):
    def __init__(self):
        super(ChannelWiseSoftmax, self).__init__()

    def forward(self, input, dim, scaling):
        return F.softmax(scaling * input, dim=1)