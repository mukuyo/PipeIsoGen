import os
import numpy as np
import cv2
import json

from numpy import ndarray

class Pipe:
    def __init__(self, name: str, num: int, pose: ndarray, vectors: list) -> None:
        self.__name = name
        self.__num = num
        self.__vectors = vectors
        self.__r_matrix: ndarray = pose[:, :3]
        self.__t_matrix: ndarray = pose[:, 3:4]

    def __str__(self) -> str:
        vectors_str = '\n'.join(map(str, self.__vectors))
        r_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__r_matrix])
        t_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__t_matrix])

        return (f"Pipe Name: {self.__name}\n"
                f"Number: {self.__num}\n"
                f"Vectors:\n{vectors_str}\n"
                f"Rotation Matrix (R):\n{r_matrix_str}\n"
                f"Translation Matrix (T):\n{t_matrix_str}")

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
    
    def compute_piping_relationship(self) -> None:
        """Compute piping relationship"""
        self.__logger.info("Start computing piping relationship")
        image = cv2.imread(self.__args.rgb_path)
        
        # Load camera parameters from JSON file
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)
        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
        
        # Coefficient to scale the length of arrows
        arrow_length = 10  # Adjust this value to change the length of the arrow
        
        pipe_count = -1

        # Load the poses of all pipes and store them in a dictionary
        for obj_name in self.__args.objects_name:
            pose_path = os.path.join(self.__args.pose_dir, obj_name, "pose.npy")
            pose_list = np.load(pose_path, allow_pickle=True)  # Load the list
            
            direction_list = [1, 2]
            if obj_name == 'tee':
                direction_list = [1, 2, -2]
            
            for i, pose_matrix in enumerate(pose_list):
                pipe_count = pipe_count + 1
                vectors = []
                for direction in direction_list:
                    if direction == -2:
                        axis_vector = -pose_matrix[:3, 2]
                    else:
                        axis_vector = pose_matrix[:3, direction]
                        
                    translation = pose_matrix[:3, 3]  # Translation vector

                    axis_end_point_3d = translation - axis_vector * arrow_length
                    
                    # Extend to the camera coordinate system to convert 3D coordinates to 2D image coordinates
                    start_point_3d = np.append(translation, 1)  # Center point after correction
                    end_point_3d = np.append(axis_end_point_3d, 1)  # Arrow tip
                    
                    # Transform using the camera matrix
                    start_point_2d_homogeneous = camera_matrix @ start_point_3d[:3]
                    end_point_2d_homogeneous = camera_matrix @ end_point_3d[:3]

                    # Normalize to convert to 2D coordinates
                    start_point_2d = (start_point_2d_homogeneous / start_point_2d_homogeneous[2])[:2]
                    end_point_2d = (end_point_2d_homogeneous / end_point_2d_homogeneous[2])[:2]
                    
                    # Convert to image coordinates
                    start_point = (int(start_point_2d[0]), int(start_point_2d[1]))
                    end_point = (int(end_point_2d[0]), int(end_point_2d[1]))

                    # Draw the center of the object for debugging (red dot)
                    cv2.circle(image, start_point, 2, (0, 0, 255), -1)  # Red dot

                    # Draw the opposite side of the Z-axis direction vector on the image
                    cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 3)  # Red arrow
                    
                    # Calculate the vector and store it
                    # vector = end_point_3d - start_point_3d
                    vectors.append(axis_vector)

                pipe = Pipe(name=obj_name, num=pipe_count, pose=pose_matrix, vectors=vectors)
                print(pipe)
                print()

        # Save the image
        output_path = 'output_image.png'
        cv2.imwrite(output_path, image)
        self.__logger.info(f"Output image saved to {output_path}")
