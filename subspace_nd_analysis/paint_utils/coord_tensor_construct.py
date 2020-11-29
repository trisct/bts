import torch

def coord_tensor_construct(N, H, W):
    coord_tensor = torch.zeros(H, W, 2)
    for y in range(H):
        for x in range(W):
            coord_tensor[y, x] = torch.tensor([x, y])
    coord_tensor = coord_tensor.unsqueeze(dim=0).expand(N, H, W, 2)
    return coord_tensor