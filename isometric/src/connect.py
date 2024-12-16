import os
import numpy as np
import cv2
import json
import math

from isometric.common.pipe import Pipe
from isometric.common.pare import Pare

from pyrealsense2 import rs2_deproject_pixel_to_point, intrinsics, distortion  # pylint: disable = no-name-in-module

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
        self.__angle_threshold = 18  # Angle threshold in degrees for determining if pipes are facing each other
        self.__relationship_loop_count = 5  # Number of angles to consider for each pipe

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__intrinsics = intrinsics()
        self.__intrinsics.width = 640  # depth image width
        self.__intrinsics.height = 480  # depth image height
        self.__intrinsics.fx = camera_matrix[0, 0]   # fx
        self.__intrinsics.fy = camera_matrix[1, 1]   # fy
        self.__intrinsics.ppx = camera_matrix[0, 2]  # cx
        self.__intrinsics.ppy = camera_matrix[1, 2]  # cy
        self.__intrinsics.model = distortion.none
        self.__intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    def find_first_pipe(self, pipes: list[Pipe]):
        """Compute piping relationship and find the first pipe"""
        self.__logger.info("Start computing to find the first pipe")
        
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
        
        bottom_left_pipe.is_first = True

        return bottom_left_pipe
    
    def get_distance(self, pipe1: Pipe, pipe2: Pipe, depth_path) -> float:
        """compute distance between two pipes"""
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth1 = float(depth_image[int(pipe1.point_2d.y), int(pipe1.point_2d.x)])
        depth2 = float(depth_image[int(pipe2.point_2d.y), int(pipe2.point_2d.x)])

        point1 = rs2_deproject_pixel_to_point(self.__intrinsics, [pipe1.point_2d.x, pipe1.point_2d.y], depth1)
        point2 = rs2_deproject_pixel_to_point(self.__intrinsics, [pipe2.point_2d.x, pipe2.point_2d.y], depth2)

        distance = np.linalg.norm(np.array(point1) - np.array(point2)) * 10.0
        # return distance
    
        _distance = np.linalg.norm(pipe1.t_matrix - pipe2.t_matrix) * 10.0
        return _distance
        return (distance + _distance) / 2
    
    def traverse_pipes(self, pipes: list[Pipe], pipe: Pipe, visited=None, edges=None):
        if visited is None:
            visited = set()
        if edges is None:
            edges = set()  # 訪問済みエッジを管理

        result = []

        # 現在のパイプを訪問済みにする
        visited.add(pipe.num)

        for pare in pipe.pare_list:
            next_pipe_num = pare.num
            edge = (pipe.num, next_pipe_num)
            reverse_edge = (next_pipe_num, pipe.num)  # 逆向きのエッジ

            # エッジが未訪問かつ逆向きのエッジも存在しない場合のみ追加
            if edge not in edges and reverse_edge not in edges:
                edges.add(edge)  # エッジを訪問済みとする
                result.append(edge)
                result.extend(self.traverse_pipes(pipes, pipes[next_pipe_num], visited, edges))

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
                for k in range(self.__relationship_loop_count):
                    if len(pare_list) + len(remain) == i + 1:
                        continue
                    pare_num = -1
                    distance_min = float('inf')

                    # 現在のパイプの方向ベクトル
                    vector = pipe.pose_matrix[:3, 2] if direction == -2 else -pipe.pose_matrix[:3, direction]

                    for other_pipe in pipes:
                        if pipe.num == other_pipe.num:
                            continue
                        for j, other_direction in enumerate(other_pipe.direction_list):
                            # 他のパイプの方向ベクトル
                            _vector = other_pipe.pose_matrix[:3, 2] if other_direction == -2 else -other_pipe.pose_matrix[:3, other_direction]

                            # 他のパイプの位置ベクトル
                            other_translation = other_pipe.pose_matrix[:3, 3]

                            # 現在のパイプの位置ベクトル
                            translation = pipe.pose_matrix[:3, 3]

                            # パイプ間の位置差ベクトル
                            relative_position = other_translation - translation
                            distance = np.linalg.norm(relative_position)

                            # 現在のパイプの方向ベクトルと位置差ベクトルの角度
                            angle = self.calculate_angle_between_vectors(vector, relative_position)

                            # 他のパイプの方向ベクトルと位置差ベクトルの角度
                            _angle = self.calculate_angle_between_vectors(_vector, -relative_position)

                            # 両方の角度が閾値以下かを判定
                            
                            angle_threshold = self.__angle_threshold + (k*5)
                            
                            if abs(angle) < angle_threshold and abs(_angle) < angle_threshold:
                                if distance_min > distance:  # 距離が最小のものを選択
                                    distance_min = distance
                                    pare_num = other_pipe.num

                    if not distance_min == float('inf') and not pare_num == -1:
                        pare_list.append(Pare(pare_num))
                        relationship.append(pipe.direction_str[i])
                    else:
                        if k == self.__relationship_loop_count - 1:
                            remain.append(pipe.direction_str[i])

            pipe.pare_list = pare_list
            pipe.relationship = relationship
            pipe.remain_relationship = remain
