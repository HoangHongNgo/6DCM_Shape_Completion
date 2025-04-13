import open3d as o3d
import numpy as np


def visualize_pc(point_cloud_tensor):
    # Convert PyTorch tensor to numpy array
    point_cloud_np = point_cloud_tensor[0, :, :].squeeze(
        0).cpu().detach().numpy()

    # Create an Open3D PointCloud object
    point_cloud_o3d = o3d.geometry.PointCloud()

    # Set the points in the Open3D PointCloud object
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)

    # Optionally, add colors to the points
    colors = np.random.rand(1024, 3)  # Random colors for each point
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud_o3d])
