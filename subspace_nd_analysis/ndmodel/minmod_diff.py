import torch
import torch.nn as nn
import torch.nn.functional as F

class MinModDiff(nn.Module):
    def __init__(self):

        super(MinModDiff, self).__init__()
        
        diff_kernels = {}
        diff_kernels['x+'] = torch.tensor([[[
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0.,-1., 0., 0., 1.],
                                [ 0., 0., 0.,-1., 0., 0., 1.],
                                [ 0., 0., 0.,-1., 0., 0., 1.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.]]]]) / 9
        diff_kernels['x-'] = torch.tensor([[[
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ -1., 0., 0., 1., 0., 0., 0.],
                                [ -1., 0., 0., 1., 0., 0., 0.],
                                [ -1., 0., 0., 1., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.]]]]) / 9
        diff_kernels['y+'] = torch.tensor([[[
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0.,-1.,-1.,-1., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 1., 1., 1., 0., 0.]]]]) / 9
        diff_kernels['y-'] = torch.tensor([[[
                                [ 0., 0.,-1.,-1.,-1., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 1., 1., 1., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.],
                                [ 0., 0., 0., 0., 0., 0., 0.]]]]) / 9

        self.pad = torch.nn.ReplicationPad2d(padding=3)
        self.diff_kernel = nn.Parameter(torch.cat(tuple(diff_kernels.values()), dim=0).expand(len(diff_kernels), 1, 7, 7))

    def forward(self, input, mode='minmod'):
        #if self.diff_kernel.device != input.device:
        #    self.diff_kernel = self.diff_kernel.to(input.device)

        N, _, H, W = input.shape
        padded_map = self.pad(input)
        diff_map = F.conv2d(padded_map, self.diff_kernel)

        if mode != 'minmod':
            return diff_map
        diff_map = {'x+': diff_map[:, 0:1],
                    'x-': diff_map[:, 1:2],
                    'y+': diff_map[:, 2:3],
                    'y-': diff_map[:, 3:4]}
        min_mod_mask = {'x+': diff_map['x+'].abs() < diff_map['x-'].abs(),
                        'y+': diff_map['y+'].abs() < diff_map['y-'].abs()}
        #print(min_mod_mask['x+'].shape)
        
        min_mod_diff = torch.zeros(N, 2, H, W, device=input.device)
        min_mod_diff[:, 0:1][ min_mod_mask['x+']] = diff_map['x+'][ min_mod_mask['x+']]
        min_mod_diff[:, 0:1][~min_mod_mask['x+']] = diff_map['x-'][~min_mod_mask['x+']]
        min_mod_diff[:, 1:2][ min_mod_mask['y+']] = diff_map['y+'][ min_mod_mask['y+']]
        min_mod_diff[:, 1:2][~min_mod_mask['y+']] = diff_map['y-'][~min_mod_mask['y+']]

        return min_mod_diff