import os
import argparse
import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def transform_points(points, rotation_matrix, translation_vector):
    rotation_matrix = np.array(rotation_matrix)
    translation_vector = np.array(translation_vector)
    return np.dot(points, rotation_matrix.T) + translation_vector

def compute_add(true_points, estimated_points):
    distances = np.linalg.norm(true_points - estimated_points, axis=1)
    return np.mean(distances)

def compute_diameter(points):
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    diameter = np.max(max_values - min_values)
    return diameter

def load_est_pose_npy(file_path):
    pose = np.load(file_path)
    rotation_matrix = pose[0][:3, :3]
    translation_vector = pose[0][:3, 3]/100.0
    return rotation_matrix, translation_vector

def load_true_pose_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    rotation = data['rotation']
    translation = data['translation']
    
    rotation_matrix = np.array([
        [rotation['row0']['x'], rotation['row0']['y'], rotation['row0']['z']],
        [rotation['row1']['x'], rotation['row1']['y'], rotation['row1']['z']],
        [rotation['row2']['x'], rotation['row2']['y'], rotation['row2']['z']]
    ])
    
    translation_vector = np.array([translation['x'], translation['y'], translation['z']])
    
    return rotation_matrix, translation_vector

def rotation_matrix_to_euler_angles(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

def main():
    parser = argparse.ArgumentParser(description="Pose Estimation")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_dir", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--true_pose_json", nargs="?", help="Path to true pose JSON file")
    args = parser.parse_args()

    object_name = 'elbow'
    model_points = load_ply(os.path.join(args.cad_dir, object_name + ".ply"))
    
    true_rotation_matrix, true_translation_vector = load_true_pose_json(os.path.join(args.output_dir, "pose", object_name, "true_pose.json"))
    estimated_rotation_matrix, estimated_translation_vector = load_est_pose_npy(os.path.join(args.output_dir, "pose", object_name, "pose.npy"))
    
    print("True Pose:")
    print("Rotation matrix:\n", true_rotation_matrix)
    print("Translation vector:\n", true_translation_vector)
    
    print("Estimated Pose:")
    print("Rotation matrix:\n", estimated_rotation_matrix)
    print("Translation vector:\n", estimated_translation_vector)
    
    true_euler_angles = rotation_matrix_to_euler_angles(true_rotation_matrix)
    estimated_euler_angles = rotation_matrix_to_euler_angles(estimated_rotation_matrix)
    
    print("True Euler Angles (degrees):", true_euler_angles)
    print("Estimated Euler Angles (degrees):", estimated_euler_angles)
    
    transformed_true_points = transform_points(model_points, true_rotation_matrix, true_translation_vector)
    transformed_estimated_points = transform_points(model_points, estimated_rotation_matrix, estimated_translation_vector)
    
    add = compute_add(transformed_true_points, transformed_estimated_points)
    diameter = compute_diameter(model_points)
    
    print(f"ADD: {add}")
    print(f"Object Diameter: {diameter} meters")
    
    if add <= 0.1 * diameter:
        print("ADD is within 10% of the object diameter, indicating high accuracy.")
    else:
        print("ADD exceeds 10% of the object diameter, indicating potential issues with accuracy.")

if __name__ == "__main__":
    main()
