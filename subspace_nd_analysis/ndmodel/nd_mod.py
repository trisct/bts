import torch
import torch.nn as nn
import torch.nn.functional as F

from ndmodel.minmod_diff import MinModDiff
from ndmodel.pixel_norm import PixelNorm

#from utils.paint import paint_true_depth

class NormDiff(nn.Module):
    def __init__(self, nd_const):

        super(NormDiff, self).__init__()

        self.minmod_diff = MinModDiff()
        self.pixel_norm = PixelNorm(nd_const)

    def forward(self, input):
        valid_bmask = input != 0

        depth = torch.zeros_like(input, device=input.device)
        depth[valid_bmask] = input[valid_bmask]
        depth[~valid_bmask] = -100.

        diff_map = self.minmod_diff(depth)
        invd_bmask = diff_map.abs() > 10.
        print(invd_bmask.shape)
        nd_map = self.pixel_norm(diff_map, dim=1)

        nd_final = torch.zeros_like(nd_map, device=nd_map.device)
        diff_final = torch.zeros_like(diff_map, device=diff_map.device)

        #paint_true_depth(inv_bmask[0,0:1].float())

        nd_final[(~invd_bmask) & valid_bmask] = nd_map[(~invd_bmask) & valid_bmask]
        diff_final[(~invd_bmask) & valid_bmask] = diff_map[(~invd_bmask) & valid_bmask]

        nd_final = F.avg_pool2d(input=nd_final, stride=1, kernel_size=5, padding=2)

    
        return nd_final, diff_final, (invd_bmask | (~valid_bmask))