import os
import numpy as np
import cv2
import json
import math

from isometric.common.pipe import Pipe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger, pipes) -> None:
        self.__args = args
        self.__logger = logger
        self.__pipes: list[Pipe] = pipes
        self.__angle_threshold = 10.0  # Angle threshold in degrees for determining if pipes are facing each other
    
    def compute_piping_relationship(self) -> None:
        """Compute piping relationship"""
        self.__logger.info("Start computing piping relationship")

        for pipe in self.__pipes:
            translation = pipe.pose_matrix[:3, 3]  # パイプの位置ベクトル
            for vector in pipe.vectors:
                distance_min = float('inf')
                for other_pipe in self.__pipes:
                    if pipe.num == other_pipe.num:
                        continue
                    
                    other_translation = other_pipe.pose_matrix[:3, 3]  # 他のパイプの位置ベクトル
                    
                    # パイプ間の位置差ベクトルを計算
                    relative_position = other_translation - translation
                    
                    distance = np.linalg.norm(relative_position)

                    # 方向ベクトル同士の角度を計算
                    angle = self.calculate_angle_between_vectors(vector, relative_position)
                    
                    # 向かい合っているかを判断
                    if abs(angle) < self.__angle_threshold:
                        if distance_min > distance:
                            distance_min = distance
                            pare_num = other_pipe.num
                    
                if not distance_min == float('inf'):
                    self.__logger.info(
                        f"{pipe.name} Pipe {pipe.num} is facing Pipe {pare_num}"
                    )

    def calculate_angle_between_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate the angle between two vectors in degrees"""
        # 正規化してから内積を計算
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 180.0  # 一方でもゼロベクトルなら180度
        
        # 正規化ベクトルを計算
        unit_vector1 = vector1 / norm1
        unit_vector2 = vector2 / norm2
        
        # 内積を計算
        dot_product = np.dot(unit_vector1, unit_vector2)
        
        # 角度をラジアンで計算し、度に変換
        angle_rad = math.acos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = angle_rad * (180.0 / math.pi)
        
        return angle_deg

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
