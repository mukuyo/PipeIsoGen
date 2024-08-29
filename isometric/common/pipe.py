import json
import numpy as np
from numpy import ndarray
from ezdxf.math import Vec3

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


class Pare:
    def __init__(self, num: int, relationship: str = "", distance: float = 0.0):
        self.__relationship: str = relationship
        self.__num: int = num
        self.__distance: float = distance

    def __str__(self) -> str:
        return (f"Pare(num: {self.__num}, "
                f"relationship: '{self.__relationship}', "
                f"distance: {self.__distance:.2f})")
    
    @property
    def relationship(self) -> str:
        return self.__relationship
    
    @relationship.setter
    def relationship(self, relationship: str) -> None:
        self.__relationship = relationship

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

        self.__center: Point = Point(0, 0, 0)

        self.__point: Point = Point(0, 0, 0)

        self.__cam_matrix: ndarray = cam_matrix

        self.__init_info()

    def __init_info(self) -> None:
        direction_list = [1, 2] if self.__name != 'tee' else [1, 2, -2]
        for direction in direction_list:
            self.__vectors.append(self.__pose_matrix[:3, 2] if direction == -2 else -self.__pose_matrix[:3, direction])
        
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__cam_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)

        self.__project_to_2d()

    def __project_to_2d(self) -> None:
        projected_point = self.__cam_matrix @ self.__t_matrix

        x = projected_point[0, 0] / projected_point[2, 0]
        y = projected_point[1, 0] / projected_point[2, 0]

        self.__center.x = int(x)
        self.__center.y = int(y)

    def __str__(self) -> str:
        vectors_str = '\n'.join(map(str, self.__vectors))
        
        r_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__r_matrix])
        
        t_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__t_matrix])

        pare_list_str = ', '.join([str(pare) for pare in self.__pare_list])  # パレリストを文字列に変換

        return (f"Pipe Name: {self.__name}\n"
                f"Number: {self.__num}\n"
                f"Vectors:\n{vectors_str}\n"
                f"Rotation Matrix (R):\n{r_matrix_str}\n"
                f"Translation Matrix (T):\n{t_matrix_str}\n"
                f"Pare List: [{pare_list_str}]\n"
                f"Center (Projected Point): {self.__center.x}, {self.__center.y}")

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

    @property
    def center(self) -> Point:
        return self.__center

    @center.setter
    def center(self, center) -> None:
        self.__center = center
    
    @property
    def point(self) -> Point:
        return self.__point

    @point.setter
    def point(self, point) -> None:
        self.__point = point

    @pare_list.setter
    def pare_list(self, pare_list: list[Pare]) -> None:
        self.__pare_list = pare_list
