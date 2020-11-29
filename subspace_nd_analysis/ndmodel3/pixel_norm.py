import torch
import torch.nn as nn

class PixelNormAdd(nn.Module):
    """
    this module performs a pixelwise normalization on an image tensor of the form [N, C, H, W]
    """
    def __init__(self, add_dim_const=None, epsilon=1e-8, add_dim=True):
        """
        add_dim_const: set a value as an additional-dimension component
        epsilong = 1e-8 is added to the squared norm for numerical stability, if add_dim_const != 0 then this is set to 0
        """ 
        super(PixelNormAdd, self).__init__()
        self.add_dim_const = add_dim_const if add_dim_const is not None else epsilon
        self.add_dim = add_dim

    def forward(self, input, dim):
        shape = list(input.shape)
        add_shape = shape.copy()
        
        shape[dim] += 1
        add_shape[dim]=1

        add_channel = torch.ones(add_shape, device=input.device) * self.add_dim_const
        
        
        cated = torch.cat((input, add_channel), dim=dim)
        #print(cated.shape, shape)
        norm = (cated ** 2).sum(dim=dim, keepdim=True)
        norm = norm.sqrt()
        norm = norm.expand(shape)
        
        output = cated / norm
        return output