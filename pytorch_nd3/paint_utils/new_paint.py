import torch
import matplotlib.pyplot as plt

def paint(image_tuple, scatter_tuple, images_per_row=3, to_screen=True, filename=None):
    """
    images are expected to be of the form [C, H, W]
    scatters are expected to be of the form [N_points, 2], where 0 at dim 2 denotes H-coordinates

    to_screen: paint to screen
    to_file: if False, it does not save to file, otherwise pass the filename as a string to this parameter
    """

    assert len(image_tuple) == len(scatter_tuple)

    plt.clf()
    n_rows = len(image_tuple) // images_per_row + 1

    if n_rows == 1:
        images_per_row = len(image_tuple)
    
    for i in range(len(image_tuple)):
        plt.subplot(n_rows, images_per_row, i+1)

        image = image_tuple[i].transpose(0,2).transpose(0,1).detach().cpu().clone()
        H, W, C = image.shape
        if C == 1:
            image = image.expand(H, W, 3).clone()
            #print(type(image))
            image /= image.max()
        plt.imshow(image)
        plt.scatter(scatter_tuple[i][:,1].cpu(), scatter_tuple[i][:,0].cpu(), s=2)
    if filename is not False:
        plt.savefig(filename)
    if to_screen:
        plt.show()

def paint_true_depth(depth_tensor):
    plt.imshow(depth_tensor.transpose(0,2).transpose(0,1))
    plt.show()