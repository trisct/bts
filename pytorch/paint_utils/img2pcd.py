import torch
import open3d as o3d

def get_coord_map(H, W):
    """
    [2, H, W]
    where
    [0, H, W] is x coord
    """
    x_axis = 2 * torch.linspace(0, W-1, steps=W) / W - 1
    y_axis = 2 * torch.linspace(0, H-1, steps=H) / H - 1
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis)
    x_grid = x_grid.expand(1, -1, -1)
    y_grid = y_grid.expand(1, -1, -1)
    coord_map = torch.cat((x_grid, -1*y_grid), dim=0)
    return coord_map

def img2pcd(*img_list, scaling=1):
    C, H, W = img_list[0].shape
    assert C == 1

    pcd_obj_list = []

    for img in img_list:
        img = img.clone().detach().cpu() * scaling

        coord_map = get_coord_map(H, W)
        pcd_points = torch.cat((coord_map, img), dim=0)
        pcd_points = pcd_points.reshape(3, -1).transpose(0,1).detach().cpu()

        #zeros = torch.zeros(600, 1)
        #axis = torch.linspace(0, 599, 600).unsqueeze(dim=1)

        #x_axis = torch.cat((axis, zeros, zeros), dim=1)
        #y_axis = torch.cat((zeros, axis, zeros), dim=1)
        #z_axis = torch.cat((zeros, zeros, axis), dim=1)

        #pcd_points = torch.cat((pcd_points, x_axis, y_axis, z_axis), dim=0)

        pcd_points = pcd_points.numpy()

        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pcd_points)
        pcd_obj_list.append(pcd_obj)

    o3d.visualization.draw_geometries(pcd_obj_list)
    #return pcd_obj, pcd_points

    