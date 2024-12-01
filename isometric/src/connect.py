import os
import numpy as np
import cv2
import json
import math

from isometric.common.pipe import Pipe, Pare

# from pyrealsense2 import rs2_deproject_pixel_to_point, intrinsics, distortion  # pylint: disable = no-name-in-module

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
        self.__angle_threshold = 25.0  # Angle threshold in degrees for determining if pipes are facing each other

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

    def find_first_pipe(self, pipes: list[Pipe]):
        """Compute piping relationship and find the first pipe"""
        self.__logger.info("Start computing piping relationship")
        
        # Initialize variables to track the most bottom-left pipe
        bottom_left_pipe = None
        min_x = float('inf')

        for pipe in pipes:
            # Check if this pipe is more bottom-left than the current one
            if pipe.point_2d.x < min_x:
                bottom_left_pipe = pipe
                min_x = pipe.point_2d.x

        if bottom_left_pipe is not None:
            self.__logger.info(f"The most bottom-left pipe is {bottom_left_pipe.name} with point_2d at {bottom_left_pipe._Pipe__point_2d}")
        else:
            self.__logger.info("No pipes found.")

        return bottom_left_pipe
    
    def get_distance(self, pipe1: Pipe, pipe2: Pipe, depth_path) -> float:
        """compute distance between two pipes"""
        distance = np.linalg.norm(pipe1.t_matrix - pipe2.t_matrix) * 10.0
        return distance
    
    def traverse_pipes(self, pipes: list[Pipe], pipe: Pipe, visited=None):
        if visited is None:
            visited = set()

        if pipe.num in visited:
            return []

        visited.add(pipe.num)
        
        result = []
        for pare in pipe.pare_list:
            next_pipe_num = pare.num
            if next_pipe_num not in visited:
                result.append((pipe.num, next_pipe_num))
                result.extend(self.traverse_pipes(pipes, pipes[next_pipe_num], visited))
    
        return result

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

    def compute_piping_relationship(self, pipes: list[Pipe]) -> None:
        """Compute piping relationship"""
        self.__logger.info("Start computing piping relationship")
        
        for pipe in pipes:
            pare_list = []
            relationship = []
            remain = []
            translation = pipe.pose_matrix[:3, 3]  # パイプの位置ベクトル
            for i, direction in enumerate(pipe.direction_list):
                pare_num = -1
                distance_min = float('inf')
                
                vector = pipe.pose_matrix[:3, 2] if direction == -2 else -pipe.pose_matrix[:3, direction]
                
                for other_pipe in pipes:
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
                    
                if not distance_min == float('inf') and not pare_num == -1:
                    pare_list.append(Pare(pare_num))
                    relationship.append(pipe.direction_str[i])
                else:
                    remain.append(pipe.direction_str[i])

            pipe.pare_list = pare_list
            pipe.relationship = relationship
            pipe.remain_relationship = remain
