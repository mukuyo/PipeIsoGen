import json
import numpy as np
from numpy import ndarray
from math import degrees
from .rotate import Rotate
from .pare import Pare
from .point import Point

class Pipe:
    def __init__(self, args, logger, obj_name, number, pose_matrix: ndarray, cam_matrix: ndarray) -> None:
        self.__args = args
        self.__logger = logger
        self.__name = obj_name
        self.__num = number
        self.__pose_matrix = pose_matrix

        self.__vectors = []
        self.__start_point_2d = []
        self.__end_point_2d = []

        self.__r_matrix: ndarray = self.__pose_matrix[:, :3]
        self.__t_matrix: ndarray = self.__pose_matrix[:, 3:4]

        self.__pare_list: list[Pare] = []

        self.__point_cad: Point = Point(0, 0)
        self.__point_2d: Point = Point(0, 0)
        self.__point_3d: Point = Point(0, 0, 0)

        self.__cam_matrix: ndarray = cam_matrix

        self.__relationship: list[str] = []
        self.__remain_relationship: list[str] = []

        self.__rotate: Rotate = Rotate(0.0, 0.0, 0.0)

        self.__is_first: bool = False

        self.__arrow_length = 10

        self.__init_info()

    def __init_info(self) -> None:
        self.__point_3d.x = self.__t_matrix[0]
        self.__point_3d.y = self.__t_matrix[1]
        self.__point_3d.z = self.__t_matrix[2]

        self.__rotate.roll = degrees(np.arctan2(self.__r_matrix[2, 1], self.__r_matrix[2, 2]))
        self.__rotate.pitch = degrees(np.arcsin(-self.__r_matrix[2, 0]))
        self.__rotate.yaw = degrees(np.arctan2(self.__r_matrix[1, 0], self.__r_matrix[0, 0]))

        if self.__name == "elbow":
            self.__direction_list = [1, 2]
        else:
            self.__direction_list = [1, 2, -2]
        self.__direction_str = ['forward', 'under', 'upper']

        self.__candidate_num: int = len(self.__direction_list)
        
        for i, direction in enumerate(self.__direction_list):
            self.__vectors.append(self.__pose_matrix[:3, 2] if direction == -2 else -self.__pose_matrix[:3, direction])

        self.__decide_direction()

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__cam_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__project_to_2d()

    def __decide_direction(self) -> None:
        min_direction = 0
        max_direction = 0
        _max_direction = 0

        pixel_distance_list = []
        
        for i, vector in enumerate(self.__vectors):
            translation = self.__pose_matrix[:3, 3]
            
            axis_end_point_3d = translation + vector * self.__arrow_length

            start_point_3d = np.append(translation, 1)  # Center point after correction
            end_point_3d = np.append(axis_end_point_3d, 1)  # Arrow tip
            
            # Transform using the camera matrix
            start_point_2d_homogeneous = self.__cam_matrix @ start_point_3d[:3]
            end_point_2d_homogeneous = self.__cam_matrix @ end_point_3d[:3]

            # Normalize to convert to 2D coordinates
            start_point_2d = (start_point_2d_homogeneous / start_point_2d_homogeneous[2])[:2]
            end_point_2d = (end_point_2d_homogeneous / end_point_2d_homogeneous[2])[:2]

            self.__start_point_2d.append((int(start_point_2d[0]), int(start_point_2d[1])))
            self.__end_point_2d.append((int(end_point_2d[0]), int(end_point_2d[1])))

            pixel_distance = start_point_2d[1] - end_point_2d[1]

            pixel_distance_list.append((i, pixel_distance))
        
        pixel_distance_list.sort(key=lambda x: x[1])
        min_direction = pixel_distance_list[0][0]
        max_direction = pixel_distance_list[-1][0]
        _max_direction = pixel_distance_list[-2][0]
        
        if self.__name == 'elbow' and min_direction == 0:
            self.__direction_list = [2, 1]
        
        if self.__name == 'tee' and max_direction == 0:
            
            if _max_direction == 2:
                self.__direction_str = ['upper', 'rforward', 'lforward']
                self.__direction_list = [1, -2, 2]
            elif _max_direction == 1:
                self.__direction_str = ['upper', 'lforward', 'rforward']
                self.__direction_list = [1, 2, -2]
        elif self.__name == 'tee' and min_direction == 0:
            self.__direction_str = ['under', 'rforward', 'lforward']
        elif self.__name == 'tee' and min_direction == 2:
            self.__direction_list = [1, -2, 2]
                
    def __project_to_2d(self) -> None:
        projected_point = self.__cam_matrix @ self.__t_matrix

        x = projected_point[0, 0] / projected_point[2, 0]
        y = projected_point[1, 0] / projected_point[2, 0]

        self.__point_2d.x = int(x)
        self.__point_2d.y = int(y)

    def __str__(self) -> str:
        vectors_str = '\n'.join(map(str, self.__vectors))
        
        r_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__r_matrix])
        
        t_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__t_matrix])

        pare_list_str = ', '.join([str(pare) for pare in self.__pare_list])  # パレリストを文字列に変換

        return (f"Pipe Name: {self.__name}\n"
                f"Number: {self.__num}\n"
                f"Relationship: {self.__relationship}\n"
                f"Vectors:\n{vectors_str}\n"
                f"Rotation Matrix (R):\n{r_matrix_str}\n"
                f"Translation Matrix (T):\n{t_matrix_str}\n"
                f"Pare List: [{pare_list_str}]\n"
                f"Center (Projected Point): {self.__point_2d.x}, {self.__point_2d.y}\n"
                f"Center (3 dimation coodinates): {self.__point_3d.x}, {self.__point_3d.y}, {self.__point_3d.z}")

    @property
    def pose_matrix(self) -> ndarray:
        return self.__pose_matrix

    @property
    def t_matrix(self) -> ndarray:
        return self.__t_matrix

    @property
    def r_matrix(self) -> ndarray:
        return self.__r_matrix

    @property
    def vectors(self) -> list:
        return self.__vectors
    
    @property
    def direction_list(self) -> list:
        return self.__direction_list

    @property
    def direction_str(self) -> list:
        return self.__direction_str

    @property
    def name(self) -> str:
        return self.__name

    @property
    def num(self) -> int:
        return self.__num
    
    @property
    def rotate(self) -> Rotate:
        return self.__rotate
    
    @property
    def is_first(self) -> bool:
        return self.__is_first
    
    @is_first.setter
    def is_first(self, is_first: bool) -> None:
        self.__is_first = is_first
    
    @property
    def pare_list(self) -> list[Pare]:
        return self.__pare_list

    @pare_list.setter
    def pare_list(self, pare_list: list[Pare]) -> None:
        self.__pare_list = pare_list

    @property
    def point_cad(self) -> Point:
        return self.__point_cad

    @point_cad.setter
    def point_cad(self, point_cad: Point) -> None:
        self.__point_cad = point_cad

    @property
    def point_2d(self) -> Point:
        return self.__point_2d

    @point_2d.setter
    def point_2d(self, point_2d: Point) -> None:
        self.__point_2d = point_2d

    @property
    def point_3d(self) -> Point:
        return self.__point_3d

    @point_3d.setter
    def point_3d(self, point_3d: Point) -> None:
        self.__point_3d = point_3d

    @property
    def start_point_2d(self) -> list[tuple[int, int]]:
        return self.__start_point_2d

    @property
    def end_point_2d(self) -> list[tuple[int, int]]:
        return self.__end_point_2d
    
    @property
    def relationship(self) -> list[str]:
        return self.__relationship
    
    @relationship.setter
    def relationship(self, relationship: list[str]) -> None:
        self.__relationship = relationship

    @property
    def remain_relationship(self) -> list[str]:
        return self.__remain_relationship
    
    @remain_relationship.setter
    def remain_relationship(self, remain_relationship: list[str]) -> None:
        self.__remain_relationship = remain_relationship

    @property
    def candidate_num(self) -> int:
        return self.__candidate_num
    
    @candidate_num.setter
    def candidate_num(self, candidate_num: int) -> None:
        self.__candidate_num = candidate_num