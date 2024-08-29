import os
import numpy as np
import cv2
import json
import math

from isometric.common.pipe import Pipe, Pare

from pyrealsense2 import rs2_deproject_pixel_to_point, intrinsics, distortion  # pylint: disable = no-name-in-module

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
        self.__angle_threshold = 10.0  # Angle threshold in degrees for determining if pipes are facing each other
        
        self.__depth_image = cv2.imread(self.__args.depth_path, cv2.IMREAD_UNCHANGED)

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__intrinsics = intrinsics()
        self.__intrinsics.width = self.__depth_image.shape[1]  # depth image width
        self.__intrinsics.height = self.__depth_image.shape[0]  # depth image height
        self.__intrinsics.fx = camera_matrix[0, 0]  # fx
        self.__intrinsics.fy = camera_matrix[1, 1]  # fy
        self.__intrinsics.ppx = camera_matrix[0, 2]  # cx
        self.__intrinsics.ppy = camera_matrix[1, 2]  # cy
        self.__intrinsics.model = distortion.none
        self.__intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    def find_first_pipe(self, pipes: list[Pipe]):
        """Compute piping relationship and find the first pipe"""
        self.__logger.info("Start computing piping relationship")
        
        # Initialize variables to track the most bottom-left pipe
        bottom_left_pipe = None
        min_x = float('inf')

        for pipe in pipes:
            # Check if this pipe is more bottom-left than the current one
            if pipe.center.x < min_x:
                bottom_left_pipe = pipe
                min_x = pipe.center.x

        if bottom_left_pipe is not None:
            self.__logger.info(f"The most bottom-left pipe is {bottom_left_pipe.name} with center at {bottom_left_pipe._Pipe__center}")
        else:
            self.__logger.info("No pipes found.")

        return bottom_left_pipe

    def get_pare_infos(self, pipe1, pipe2):
        relationship = ""
        if abs(pipe1.center.x - pipe2.center.x) > abs(pipe1.center.y - pipe2.center.y):
            if np.sign(pipe1.center.x - pipe2.center.x) > 0:
                relationship = "left"
            else:
                relationship = "right"
        else:
            if np.sign(pipe1.center.y - pipe2.center.y) > 0:
                relationship = "upper"
            else:
                relationship = "under"
        distance = self.__compute_dist_between_pipes(pipe1, pipe2)
        return relationship, distance
    
    def __compute_dist_between_pipes(self, pipe1: Pipe, pipe2: Pipe):
        """compute distance between two pipes"""
        depth1 = float(self.__depth_image[int(pipe1.center.y), int(pipe1.center.x)])
        depth2 = float(self.__depth_image[int(pipe2.center.y), int(pipe2.center.x)])

        point1 = rs2_deproject_pixel_to_point(self.__intrinsics, [pipe1.center.x, pipe1.center.y], depth1)
        point2 = rs2_deproject_pixel_to_point(self.__intrinsics, [pipe2.center.x, pipe2.center.y], depth2)

        distance = np.linalg.norm(np.array(point1) - np.array(point2))

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
            translation = pipe.pose_matrix[:3, 3]  # パイプの位置ベクトル
            for vector in pipe.vectors:
                pare_num = -1
                distance_min = float('inf')
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
                            # if not pipe.num in other_pipe.pare_list:
                            pare_num = other_pipe.num
                    
                if not distance_min == float('inf') and not pare_num == -1:
                    pare_list.append(Pare(pare_num))
            pipe.pare_list = pare_list
