import torch
import matplotlib.pyplot as plt

def Unnormalize(img_tensor, means, stds):
    """
    takes as input an image tensor of the shape [C, H, W]
    returns a tensor of the same shape unnormalized by means and stds

    means and stds should come in tuples and match the size of the channels
    """
    # print(means, type(means))
    for i in range(len(means)):
        img_tensor[i] *= stds[i]
        img_tensor[i] += means[i]
    return img_tensor

def paint(img_tensor, unnorm=False, transp=False, to_screen=True, to_file=False):
    """
    takes as input an image tensor of the shape [C, H, W]
    transforms it into an array of the shape [H, W, C] for imshow by matplotlib
    paints the image

    params:

    unnorm: unnormalize by mean 0.5 and std 0.5 on each channel
    transp: transpose from [C, H, W] to [H, W, C]
    to_screen: paint to screen
    to_file: if False, it does not save to file, otherwise pass the filename as a string to this parameter
    """

    img_tensor = img_tensor.clone().detach().cpu()
    n_channels = img_tensor.shape[0]

    if n_channels == 1:
        means = .5,
        stds = .5,
    elif n_channels == 3:
        means = .5, .5, .5
        stds = .5, .5, .5
    else:
        print('Invalid number of channels for painting')
    if unnorm:
        img_tensor = Unnormalize(img_tensor, means, stds)
    if transp:
        img_tensor = img_tensor.transpose(0,2).transpose(0,1).squeeze()
    plt.imshow(img_tensor)
    if to_file is not False:
        plt.savefig(to_file, dpi=400, bbox_inches='tight')
    if to_screen:
        plt.show()
    plt.clf()

def paint_true_depth(depth_tensor):
    plt.imshow(depth_tensor.transpose(0,2).transpose(0,1))
    plt.show()

def paint_multiple(*images, to_screen=True, to_file=False, images_per_row=1):
    """
    takes in multiple images and put them along side each other

    either paints them to screen or file according to the parameters
    """
    final_img = concat_img(images, images_per_row=images_per_row)
    paint(final_img, False, True, to_screen=to_screen, to_file=to_file)

def concat_img(image_tuple, images_per_row):
    """
    !!! detach before passing the images in
    
    takes in a sequence of images of the same size
    outputs an image containing all of them

    all tensors representing the images are of the form [C, H, W]
    """

    C, H, W = image_tuple[0].shape

    num_images = len(image_tuple)
    num_images += ((images_per_row - (num_images % images_per_row)) % images_per_row)

    final_img = torch.zeros(3, num_images, H, W).reshape(3, -1, W * images_per_row)

    for i, ori_img in enumerate(image_tuple):
        left_top = {'x': (i%images_per_row) * W, 'y': (i//images_per_row) * H}
        if ori_img is None:
            final_img[:, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = 1.
            continue
        
        img = ori_img.clone().detach().cpu()
        if img.shape[0] == 1:
            img -= img.min()
            final_img[0, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = img[0].true_divide(img.max())
            final_img[1, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = img[0].true_divide(img.max())
            final_img[2, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = img[0].true_divide(img.max())
        if img.shape[0] == 2:
            img -= img.min()
            final_img[1:3, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = img.true_divide(img.max())
        if img.shape[0] == 3:
            img -= img.min()
            final_img[:, left_top['y']:left_top['y']+H, left_top['x']:left_top['x']+W] = img.true_divide(img.max())
    return final_img
