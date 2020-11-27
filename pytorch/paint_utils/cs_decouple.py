# This module takes in an RGB image batch and normlizes the color vectors while putting the color strength to a separate channel
import torch

def cs_decouple(img_tensor):
    """
    img_tensor: a tensor of the form [N, C, H, W]
    """
    img_tensor = img_tensor.clone().detach()
    N, H, W = img_tensor.shape[0], img_tensor.shape[2], img_tensor.shape[3]

    img_color_norm = img_tensor.norm(dim=1, keepdim=True)
    normalized_img = img_tensor / img_color_norm.expand(N, 3, H, W)
    img_with_strength = torch.cat((normalized_img, img_color_norm), dim=1)
    return img_with_strength
