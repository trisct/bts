import torch
import torch.nn as nn
import torch.nn.functional as F

from ndmodel3.minmod_diff import MinModDiff
from ndmodel3.pixel_norm import PixelNormAdd

#from utils.paint import paint_true_depth

class NormDiff(nn.Module):
    def __init__(self, nd_const):
        """
        I recommend 5e-4 for disparity maps of NYUDv2
        """

        super(NormDiff, self).__init__()

        self.minmod_diff = MinModDiff()
        self.pixel_norm = PixelNormAdd(nd_const)

    def forward(self, input):
        valid_bmask = (input != 0) # [1, 1, H, W]

        depth = torch.zeros_like(input, device=input.device)
        depth[valid_bmask] = input[valid_bmask]
        depth[~valid_bmask] = -100.

        diff_map = self.minmod_diff(depth)
        N, _, H, W = diff_map.shape

        invd_bmask = diff_map.abs() > 10.
        invd_bmask = (invd_bmask[:,0:1] | invd_bmask[:,1:2]) # [1, 1, H, W]
        
        nd_map = self.pixel_norm(diff_map, dim=1)

        nd_final = torch.zeros(N, 3, H, W, device=nd_map.device)
        diff_final = torch.zeros(N, 2, H, W, device=diff_map.device)

        #paint_true_depth(inv_bmask[0,0:1].float())

        nd_final[((~invd_bmask) & valid_bmask).expand(-1, 3,-1,-1)] = nd_map[((~invd_bmask) & valid_bmask).expand(-1, 3,-1,-1)]
        diff_final[((~invd_bmask) & valid_bmask).expand(-1, 2,-1,-1)] = diff_map[((~invd_bmask) & valid_bmask).expand(-1, 2,-1,-1)]

        nd_final = F.avg_pool2d(nd_final, kernel_size=5, stride=1, padding=2)
        nd_final = F.avg_pool2d(nd_final, kernel_size=5, stride=1, padding=2)
        nd_final = F.avg_pool2d(nd_final, kernel_size=5, stride=1, padding=2)
        #nd_final = F.avg_pool2d(nd_final, kernel_size=5, stride=1, padding=2)
        #nd_final = F.avg_pool2d(nd_final, kernel_size=5, stride=1, padding=2)

        return nd_final, diff_final, (invd_bmask | (~valid_bmask))