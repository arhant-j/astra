import open3d as o3d
import numpy as np

def generate_fake_lunar_data():
    num_points = 1000
    x = np.random.uniform(low=0, high=5, size=num_points)
    y = np.random.uniform(low=0, high=10, size=num_points)
    z = np.random.uniform(low=0, high=1, size=num_points)
    return np.column_stack((x, y, z))

def preprocess_point_cloud(point_cloud):
    # No preprocessing in this example
    return point_cloud

def perform_3d_cloud_stitching(point_cloud_ref, point_cloud_current, grid_size=0.1, merge_size=0.015):
    fixed = o3d.geometry.PointCloud()
    fixed.points = o3d.utility.Vector3dVector(point_cloud_ref)
    
    moving = o3d.geometry.PointCloud()
    moving.points = o3d.utility.Vector3dVector(point_cloud_current)
    
    # Downsample point clouds
    fixed = fixed.voxel_down_sample(voxel_size=grid_size)
    moving = moving.voxel_down_sample(voxel_size=grid_size)
    
    # Perform ICP registration
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=moving, target=fixed,
        max_correspondence_distance=0.01,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    # Transform the current point cloud to the reference coordinate system
    pt_cloud_aligned = moving.transform(reg_p2l.transformation)
    
    # Merge point clouds
    pt_cloud_scene = fixed + pt_cloud_aligned
    
    return pt_cloud_scene

def visualize_point_cloud(point_cloud, title):
    o3d.visualization.draw_geometries([point_cloud], window_name=title)

def main():
    # Step 1: Generate fake lunar data
    lunar_data = [generate_fake_lunar_data() for _ in range(3)]

    # Step 2: Preprocess the data
    pt_cloud_ref = preprocess_point_cloud(lunar_data[0])
    
    # Step 3: Perform 3D cloud stitching
    pt_cloud_scene = None
    for i in range(1, len(lunar_data)):
        pt_cloud_current = preprocess_point_cloud(lunar_data[i])
        pt_cloud_scene = perform_3d_cloud_stitching(pt_cloud_ref, pt_cloud_current)
        visualize_point_cloud(pt_cloud_scene, f"World Scene After Iteration {i}")

if __name__ == "__main__":
    main()
