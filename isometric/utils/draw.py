import os
import cv2
import numpy as np
import json
import ezdxf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, sin, cos, pi
from isometric.common.pipe import Pipe, Pare, Point
from ezdxf.math import Vec3

class DrawUtils:
    """Draw Utils class"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger

        # self.__image = cv2.imread(self.__args.image_path)
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__arrow_length = 10

        self.__doc = ezdxf.new()

        dimstyle = self.__doc.dimstyles.new('custom_dimstyle')
        dimstyle.dxf.dimtxt = 17
        dimstyle.dxf.dimdec = 0
        dimstyle.dxf.dimasz = 30
        dimstyle.dxf.dimblk = "OPEN"
        dimstyle.dxf.dimclrd = 3
        dimstyle.dxf.dimclre = 3
        
        self.__msp = self.__doc.modelspace()

    def pipe_line(self, start_pipe: Pipe, end_pipe: Pipe, distance: float):
        dx = end_pipe.point_3d.x - start_pipe.point_3d.x
        dy = end_pipe.point_3d.y - start_pipe.point_3d.y
        dz = end_pipe.point_3d.z - start_pipe.point_3d.z

        if abs(dx) > abs(dy):
            pipe_rad = pi/6 * -np.sign(dy)
            if dx < 0:
                pipe_rad = -pipe_rad + pi
            start_pipe.relationship = "forward"
            end_pipe.relationship = "forward"
        else:
            pipe_rad = pi/2 * -np.sign(dy)
            start_pipe.relationship = "under"
            end_pipe.relationship = "upper"

        start_point = Vec3(start_pipe.point_cad.x, start_pipe.point_cad.y)
        end_point = Vec3(distance*cos(pipe_rad)+start_point.x, distance*sin(pipe_rad)+start_point.y)

        self.__msp.add_line(start_point, end_point)

        self.__pipe_symbol(start_point, end_point, pipe_rad)
        # self.__msp.add_line(Vec3(start_point.x - 70*cos(pipe_rad), start_point.y - 70*sin(pipe_rad)), Vec3(start_point.x + 70*cos(pipe_rad), start_point.y - 70*sin(pipe_rad)))
        # self.__msp.add_line(Vec3(start_point.x - 10*cos(pipe_rad), start_point.y - 75*sin(pipe_rad)), Vec3(start_point.x + 10*cos(pipe_rad), start_point.y - 75*sin(pipe_rad)))
        
        # elif start_pipe.relationship == "upper":
            # self.__msp.add_line(Vec3(start_point.x - 10, start_point.y + 75), Vec3(start_point.x + 10, start_point.y + 75))
        # elif start_pipe.relationship == "forward":
        #     self.__msp.add_line(Vec3(start_point.x + , start_point.y - 75), Vec3(start_point.x + 10, start_point.y - 75))
        # self.__pipe_symbol(self, start_pipe, end_pipe)

        self.__msp.add_aligned_dim(
            p1=start_point if dx > 0 else end_point,
            p2=end_point if dx > 0 else start_point,
            distance=-30,
            dimstyle="custom_dimstyle",
            text=str(round(distance, 2))
            ).render()
        
        end_pipe.point_cad = end_point
    
    def __pipe_symbol(self, start_point: Vec3, end_point: Vec3, pipe_rad: float, remain_flag: bool = False):
        symbol_rad = pipe_rad
        if not abs(symbol_rad) == pi/2:
            symbol_rad = 0

        self.__msp.add_line(Vec3(start_point.x + 50*cos(pipe_rad) + 10*sin(symbol_rad), 
                                 start_point.y + 50*sin(pipe_rad) + 10*cos(symbol_rad)), 
                            Vec3(start_point.x + 50*cos(pipe_rad) - 10*sin(symbol_rad), 
                                 start_point.y + 50*sin(pipe_rad) - 10*cos(symbol_rad)))
        if remain_flag:
            return
        
        self.__msp.add_line(Vec3(end_point.x - 50*cos(pipe_rad) + 10*sin(symbol_rad), 
                                 end_point.y - 50*sin(pipe_rad) + 10*cos(symbol_rad)), 
                            Vec3(end_point.x - 50*cos(pipe_rad) - 10*sin(symbol_rad), 
                                 end_point.y - 50*sin(pipe_rad) - 10*cos(symbol_rad)))
        
    def remain_pipe_line(self, start_pipe: Pipe):
        # dx = end_pipe.point_3d.x - start_pipe.point_3d.x
        # dy = end_pipe.point_3d.y - start_pipe.point_3d.y
        # dz = end_pipe.point_3d.z - start_pipe.point_3d.z

        pipe_rad = -pi/2
        distance = 400
        start_point = Vec3(start_pipe.point_cad.x, start_pipe.point_cad.y)
        end_point = Vec3(distance*cos(pipe_rad)+start_point.x, distance*sin(pipe_rad)+start_point.y)

        self.__msp.add_line(start_point, end_point)

        self.__msp.add_line(Vec3(end_point.x - 10, end_point.y), Vec3(end_point.x + 10, end_point.y))

        self.__pipe_symbol(start_point, end_point, pipe_rad, remain_flag=True)
        # self.__msp.add_aligned_dim(
        #     p1=start_point,
        #     p2=end_point,
        #     distance=-20,
        #     dimstyle="custom_dimstyle",
        #     ).render()

    def pipe_direction(self, pipes: list[Pipe], save_dir, img_num: int) -> None:
        self.__image = cv2.imread(os.path.join(self.__args.img_dir, f"rgb/frame{str(img_num)}.png"))
        
        for pipe in pipes:
            for i, vector in enumerate(pipe.vectors):
                if i == 0:
                    color = (255, 0, 0)  # Green color
                elif i == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                translation = pipe.pose_matrix[:3, 3]
                # translation = np.array([15.467042922973633, -38.80397033691406, -276.7315979003906])

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
                cv2.circle(self.__image, start_point, 2, color, -1)  # Red dot

                # Draw the opposite side of the Z-axis direction vector on the image
                cv2.arrowedLine(self.__image, start_point, end_point, color, 2)  # Red arrow

                # Draw the pipe number
                pipe_number_text = f"{pipe.num}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5

                color = (255, 0, 0)
                thickness = 2
                text_size, _ = cv2.getTextSize(pipe_number_text, font, font_scale, thickness)
                text_x = start_point[0] - text_size[0] // 2
                text_y = start_point[1] - text_size[1] // 2
                cv2.putText(self.__image, pipe_number_text, (text_x, text_y), font, font_scale, color, thickness)

        # Save the image
        save_path = os.path.join(save_dir, "pipe_direction.png")
        cv2.imwrite(save_path, self.__image)

    def save_dxf(self, img_num) -> None:
        self.__doc.saveas(os.path.join(self.__args.output_dir, "isometric", str(img_num), "pipe.dxf"))

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