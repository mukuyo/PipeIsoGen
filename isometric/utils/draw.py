import os
import cv2
import numpy as np
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from isometric.common.pipe import Pipe

class DrawUtils:
    """Draw Utils class"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger

        self.__image = cv2.imread(self.__args.rgb_path)
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__arrow_length = 10

    def pipe_direction(self, pipes: list[Pipe]) -> None:
        self.__tmp_image = self.__image.copy()

        for pipe in pipes:
            for vector in pipe.vectors:
                translation = pipe.pose_matrix[:3, 3]
                axis_end_point_3d = translation + vector * self.__arrow_length

                # Extend to the camera coordinate system to convert 3D coordinates to 2D image coordinates
                start_point_3d = np.append(translation, 1)  # Center point after correction
                end_point_3d = np.append(axis_end_point_3d, 1)  # Arrow tip
                
                # Transform using the camera matrix
                start_point_2d_homogeneous = self.__camera_matrix @ start_point_3d[:3]
                end_point_2d_homogeneous = self.__camera_matrix @ end_point_3d[:3]

                # Normalize to convert to 2D coordinates
                start_point_2d = (start_point_2d_homogeneous / start_point_2d_homogeneous[2])[:2]
                end_point_2d = (end_point_2d_homogeneous / end_point_2d_homogeneous[2])[:2]
                
                # Convert to image coordinates
                start_point = (int(start_point_2d[0]), int(start_point_2d[1]))
                end_point = (int(end_point_2d[0]), int(end_point_2d[1]))

                # Draw the center of the object for debugging (red dot)
                cv2.circle(self.__tmp_image, start_point, 2, (0, 0, 255), -1)  # Red dot

                # Draw the opposite side of the Z-axis direction vector on the image
                cv2.arrowedLine(self.__tmp_image, start_point, end_point, (0, 0, 255), 3)  # Red arrow

                # Draw the pipe number
                pipe_number_text = f"{pipe.num}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 0, 0)  # Green color
                thickness = 2
                text_size, _ = cv2.getTextSize(pipe_number_text, font, font_scale, thickness)
                text_x = start_point[0] - text_size[0] // 2
                text_y = start_point[1] - text_size[1] // 2
                cv2.putText(self.__tmp_image, pipe_number_text, (text_x, text_y), font, font_scale, color, thickness)

        # Save the image
        save_path = os.path.join(self.__args.output_dir, "isometric/", "pipe_direction.png")
        cv2.imwrite(save_path, self.__tmp_image)
        self.__logger.info(f"Output image saved to {save_path}")

    def plot_vectors_3d(self) -> None:
        """Plot vectors in 3D space with proper origins"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for pipe in self.__pipes:
            origin = pipe.pose_matrix[:3, 3]  # Get the starting position from pipe.pose_matrix
            
            for vector in pipe.vectors:
                # Plot the vector from the origin defined by pipe.pose_matrix
                ax.quiver(
                    origin[0], origin[1], origin[2], 
                    vector[0], vector[1], vector[2], 
                    length=10.0, normalize=True
                )
                # Annotate the vector with the pipe name and number
                ax.text(
                    origin[0] + vector[0], 
                    origin[1] + vector[1], 
                    origin[2] + vector[2], 
                    f'{pipe.name} {pipe.num}', 
                    color='red'
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Pipe Vectors')

        plt.show()