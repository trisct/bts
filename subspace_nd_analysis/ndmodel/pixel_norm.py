import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    """
    this module performs a pixelwise normalization on an image tensor of the form [N, C, H, W]
    """
    def __init__(self, addi_dim_const=0, epsilon=1e-8):
        """
        addi_dim_const: set a value as an additional-dimension component
        epsilong = 1e-8 is added to the squared norm for numerical stability, if addi_dim_const != 0 then this is set to 0
        """ 
        super(PixelNorm, self).__init__()
        self.addi_dim_const = addi_dim_const if addi_dim_const != 0 else epsilon

    def forward(self, input, dim):
        shape = input.shape
        norm = (input ** 2).sum(dim=dim, keepdim=True)
        norm += (self.addi_dim_const ** 2)
        norm = torch.sqrt(norm)
        norm = norm.expand(shape)
        
        output = input / norm
        return output