import os
import cv2
import numpy as np
import json
import ezdxf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, cos, pi
from isometric.common.pipe import Pipe, Pare, Point
from ezdxf.math import Vec3

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

        self.__doc = ezdxf.new()

        dimstyle = self.__doc.dimstyles.new('custom_dimstyle')
        dimstyle.dxf.dimtxt = 30
        dimstyle.dxf.dimdec = 0
        dimstyle.dxf.dimasz = 20
        dimstyle.dxf.dimblk = "OPEN"
        dimstyle.dxf.dimclrd = 3
        dimstyle.dxf.dimclre = 3
        
        self.__msp = self.__doc.modelspace()

    def __draw_under(self, point1: Vec3, distance):
        po2 = Point(point1.x, point1.y - distance)
        point2 = Vec3(po2.x, po2.y)
        self.__msp.add_line(point1, point2)
        self.__msp.add_aligned_dim(
            p1=point1,
            p2=point2,
            distance=-25,
            dimstyle="custom_dimstyle",
            text=str(round(distance, 2))
            ).render()
        return po2

    def __draw_right(self, point1: Vec3, distance):
        # direction_radian = pi/6
        # if direction == 0:
        #     direction_radian = -pi/6
        # elif direction == 1:
        #     direction_radian = pi/6
        # elif direction == 2:
        #     direction_radian = 5*pi/6
        # elif direction == 3:
        #     direction_radian = -5*pi/6
        # point2 = (int(distance*cos(direction_radian)+point1[0]), int(distance*sin(direction_radian)+point1[1]))
        po2 = Point(distance*cos(pi/6)+point1.x, distance*cos(pi/6)+point1.y)
        point2 = Vec3(po2.x, po2.y)
        self.__msp.add_line(point1, point2)
        distance = int(sqrt((point2[1] - point1[1]) * (point2[1] - point1[1]) + (point2[0] - point1[0]) * (point2[0] - point1[0])))
        if point1[0] < point2[0]:
            self.__msp.add_aligned_dim(
                p1=point1,
                p2=point2,
                distance=-20,
                dimstyle="custom_dimstyle",
                text=str(round(distance, 2))
                ).render()
        else:
            self.__msp.add_aligned_dim(
                p1=point2,
                p2=point1,
                distance=-20,
                dimstyle="custom_dimstyle",
                text=str(round(distance, 2))
                ).render()
        return po2
    
    def pipe_line(self, origin_pipe: Pipe, next_pipe: Pipe, relationship, distance) -> None:
        if relationship == "under":
            point = self.__draw_under(Vec3(origin_pipe.point.x, origin_pipe.point.y), distance)
        elif relationship == "right":
            point = self.__draw_right(Vec3(origin_pipe.point.x, origin_pipe.point.y), distance)
        next_pipe.point = point
    
    def pipe_direction(self, pipes: list[Pipe]) -> None:
        self.__tmp_image = self.__image.copy()

        for pipe in pipes:
            for i, vector in enumerate(pipe.vectors):
                if i == 0:
                    color = (255, 0, 0)  # Green color
                elif i == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

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
                cv2.circle(self.__tmp_image, start_point, 2, color, -1)  # Red dot

                # Draw the opposite side of the Z-axis direction vector on the image
                cv2.arrowedLine(self.__tmp_image, start_point, end_point, color, 3)  # Red arrow

                # Draw the pipe number
                pipe_number_text = f"{pipe.num}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5

                color = (255, 0, 0)
                thickness = 2
                text_size, _ = cv2.getTextSize(pipe_number_text, font, font_scale, thickness)
                text_x = start_point[0] - text_size[0] // 2
                text_y = start_point[1] - text_size[1] // 2
                cv2.putText(self.__tmp_image, pipe_number_text, (text_x, text_y), font, font_scale, color, thickness)

        # Save the image
        save_path = os.path.join(self.__args.output_dir, "isometric/", "pipe_direction.png")
        cv2.imwrite(save_path, self.__tmp_image)
        self.__logger.info(f"Output image saved to {save_path}")

    def save_dxf(self) -> None:
        self.__doc.saveas(os.path.join(self.__args.output_dir, "isometric/", "pipe.dxf"))

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