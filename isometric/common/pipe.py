import json
import numpy as np
from numpy import ndarray
from ezdxf.math import Vec3
from math import degrees

class Point:
    def __init__(self, x: int, y: int, z: int = 0):
        self.__x = x
        self.__y = y
        self.__z = z

    def __str__(self) -> str:
        return f"Point(x={self.__x}, y={self.__y}, z={self.__z})"

    @property
    def x(self) -> int:
        return self.__x
    
    @x.setter
    def x(self, x: int) -> None:
        self.__x = x

    @property
    def y(self) -> int:
        return self.__y
    
    @y.setter
    def y(self, y: int) -> None:
        self.__y = y

    @property
    def z(self) -> int:
        return self.__z
    
    @z.setter
    def z(self, z: int) -> None:
        self.__z = z


class Rotate:
    def __init__(self, roll: int, pitch: int, yaw: int = 0):
        self.__roll = roll
        self.__pitch = pitch
        self.__yaw = yaw

    def __str__(self) -> str:
        return f"Point(roll={self.__roll}, pitch={self.__pitch}, yaw={self.__yaw})"

    @property
    def roll(self) -> int:
        return self.__roll
    
    @roll.setter
    def roll(self, roll: int) -> None:
        self.__roll = roll

    @property
    def pitch(self) -> int:
        return self.__pitch
    
    @pitch.setter
    def pitch(self, pitch: int) -> None:
        self.__pitch = pitch

    @property
    def yaw(self) -> int:
        return self.__yaw
    
    @yaw.setter
    def yaw(self, yaw: int) -> None:
        self.__yaw = yaw


class Pare:
    def __init__(self, num: int, distance: float = 0.0):
        self.__num: int = num
        self.__distance: float = distance

    def __str__(self) -> str:
        return (f"Pare(num: {self.__num}, "
                f"distance: {self.__distance:.2f})")
    
    @property
    def num(self) -> int:
        return self.__num
    
    @num.setter
    def num(self, num: int):
        self.__num = num

    @property
    def distance(self) -> float:
        return self.__distance
    
    @distance.setter
    def distance(self, distance: float) -> None:
        self.__distance = distance


class Pipe:
    def __init__(self, args, logger, obj_name, number, pose_matrix: ndarray, cam_matrix: ndarray) -> None:
        self.__args = args
        self.__logger = logger
        self.__name = obj_name
        self.__num = number
        self.__pose_matrix = pose_matrix
        self.__vectors = []

        self.__r_matrix: ndarray = self.__pose_matrix[:, :3]
        self.__t_matrix: ndarray = self.__pose_matrix[:, 3:4]

        self.__pare_list: list[Pare] = []

        self.__point_cad: Point = Point(0, 0)
        self.__point_2d: Point = Point(0, 0)
        self.__point_3d: Point = Point(0, 0, 0)

        self.__cam_matrix: ndarray = cam_matrix

        self.__relationship: list[str] = []

        self.__rotate: Rotate = Rotate(0.0, 0.0, 0.0)

        self.__init_info()

    def __init_info(self) -> None:
        self.__point_3d.x = self.__t_matrix[0]
        self.__point_3d.y = self.__t_matrix[1]
        self.__point_3d.z = self.__t_matrix[2]

        self.__rotate.roll = degrees(np.arctan2(self.__r_matrix[2, 1], self.__r_matrix[2, 2]))
        self.__rotate.pitch = degrees(np.arcsin(-self.__r_matrix[2, 0]))
        self.__rotate.yaw = degrees(np.arctan2(self.__r_matrix[1, 0], self.__r_matrix[0, 0]))

        if self.__name == "elbow":
            direction_list = [1, 2]
        else:
            direction_list = [1, 2, -2]
        
        self.__candidate_num: int = len(direction_list)
        
        for i, direction in enumerate(direction_list):
            self.__vectors.append(self.__pose_matrix[:3, 2] if direction == -2 else -self.__pose_matrix[:3, direction])

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__cam_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__project_to_2d()


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
    def name(self) -> str:
        return self.__name

    @property
    def num(self) -> int:
        return self.__num
    
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
    def relationship(self) -> list[str]:
        return self.__relationship
    
    @relationship.setter
    def relationship(self, relationship: str) -> None:
        self.__relationship.append(relationship)

    @property
    def candidate_num(self) -> int:
        return self.__candidate_num
    
    @candidate_num.setter
    def candidate_num(self, candidate_num: int) -> None:
        self.__candidate_num = candidate_num