import cv2
import torch
import numpy as np
import open3d as o3d


def generate_pc_from_diff(cloud, diff):
    """
    Generate Rear Point Cloud from Front Point Cloud and Front-Rear Offset Map

    Args:
        cloud: Front point cloud (B, N_point, 3)
        diff: Front-rear offset map (B, N_point, 1)
        format: Output format ('cloud_rear' or 'img_rear')

    Returns:
        Rear point cloud in specified format:
        - 'cloud_rear': (B, N_point, 3)
        - 'img_rear': Rear cloud in the same shape as input cloud (B, N_point, 3)
    """
    # Validate shapes
    assert cloud.shape[2] == 3, "cloud must have shape (B, N_point, 3)"
    assert diff.shape[2] == 1 and diff.shape[1] == cloud.shape[
        1], "diff must have shape (B, N_point, 1)"

    # Calculate unit vectors for each point in the front cloud
    uv = torch.norm(cloud, dim=2, keepdim=True)  # (B, N_point, 1)
    uv = torch.div(cloud, uv)  # Normalize cloud to unit vectors
    uv[torch.isnan(uv)] = 0  # Handle division by zero

    # Calculate the rear point cloud
    fr = uv * diff  # Offset vectors (B, N_point, 3)
    cr = cloud + fr  # Rear point cloud (B, N_point, 3)

    return cr


def ssd2pointcloud(cloud, mask, diff, format='img_rear'):
    """
    Generate Point Cloud from Front Point Cloud and FROM

    Args:
        cloud: Point cloud (from depth image)
        mask: Semantic mask
        diff: Front-rear offset map
        format: ??

    Returns:

    """
    mask_obj = np.expand_dims(np.asarray(mask > 0), axis=2)
    # Find unit vector of front view pt
    uv = np.sqrt(
        np.power(cloud[..., 0], 2) + np.power(cloud[..., 1], 2) + np.power(cloud[..., 2], 2))
    uv = cloud[..., :] / np.expand_dims(uv, axis=2)
    uv[np.isnan(uv)] = 0
    fr = uv[..., :] * np.expand_dims(diff, axis=2)
    cr = (cloud + fr) * mask_obj
    cloudm = cloud * mask_obj
    obj_pt = np.append(cr.reshape(-1, 3), cloud.reshape(-1, 3), axis=0)

    if format == 'img_rear':
        return cr
    elif format == 'cloud_rear':
        return cr.reshape(-1, 3)


"""
    for tensor processing, rewrite with pytorch 
"""


def ssd2pointcloud_tensor(cloud: torch.Tensor, mask: torch.Tensor, diff: torch.Tensor, format: str = 'img_rear'):
    """
    Generate Point Cloud from Front Point Cloud and FROM

    Args:
        cloud: Point cloud tensor of shape (batch_size, 3, H, W)
        mask: Semantic mask tensor of shape (batch_size, 1, H, W)
        diff: Front-rear offset map tensor of shape (batch_size, 1, H, W)
        format: Output format, either 'img_rear' or 'cloud_rear'

    Returns:
        Processed point cloud in the requested format
    """
    mask_obj = (mask > 0).float()

    # Compute unit vector of front view point
    # Avoid division by zero
    uv = torch.sqrt(torch.sum(cloud ** 2, dim=1, keepdim=True) + 1e-8)
    uv = cloud / uv

    fr = uv * (diff + 1e-7)  # Add small value to avoid exact zero
    cr = (cloud + fr) * mask_obj
    # cloudm = cloud * mask_obj

    # obj_pt = torch.cat([cr.view(-1, 3), cloud.view(-1, 3)], dim=0)

    if format == 'img_rear':
        return cr
    elif format == 'cloud_rear':
        return cr.view(-1, 3)
    else:
        raise ValueError(
            "Invalid format. Choose either 'img_rear' or 'cloud_rear'.")
