import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseSoftmax(nn.Module):
    def __init__(self):
        super(ChannelWiseSoftmax, self).__init__()

    def forward(self, input, dim, scaling):
        _,  C, _, _ = input.shape
        per_channel = []
        for c in range(C):
            map_pos = torch.zeros_like(input[:, c:c+1], device=input.device)
            map_neg = torch.zeros_like(input[:, c:c+1], device=input.device)
            map_pos[ input[:, c:c+1] >= 0 ] = input[:, c:c+1][ input[:, c:c+1] >= 0 ].abs()
            map_neg[ input[:, c:c+1] <  0 ] = input[:, c:c+1][ input[:, c:c+1] <  0 ].abs()
            per_channel.append(map_pos)
            per_channel.append(map_neg)
            
        cated = torch.cat(per_channel, dim=1)
        return F.softmax(scaling * cated, dim=1)