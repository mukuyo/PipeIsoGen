import numpy as np
from numpy import ndarray

class Pipe:
    def __init__(self, args, logger, obj_name, number, pose_matrix: ndarray) -> None:
        self.__args = args
        self.__logger = logger
        self.__name = obj_name
        self.__num = number
        self.__pose_matrix = pose_matrix
        self.__vectors = []

        self.__r_matrix: ndarray = self.__pose_matrix[:, :3]
        self.__t_matrix: ndarray = self.__pose_matrix[:, 3:4]

        self.__pare_list: list = []

        self.__init_info()

    def __str__(self) -> str:
        vectors_str = '\n'.join(map(str, self.__vectors))
        
        r_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__r_matrix])
        
        t_matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in self.__t_matrix])

        return (f"Pipe Name: {self.__name}\n"
                f"Number: {self.__num}\n"
                f"Vectors:\n{vectors_str}\n"
                f"Rotation Matrix (R):\n{r_matrix_str}\n"
                f"Translation Matrix (T):\n{t_matrix_str}\n"
                f"Pare List: {self.__pare_list}")

    def __init_info(self) -> None:
        direction_list = [1, 2] if self.__name != 'tee' else [1, 2, -2]
        for direction in direction_list:
            self.__vectors.append(self.__pose_matrix[:3, 2] if direction == -2 else -self.__pose_matrix[:3, direction])

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
    def pare_list(self) -> list:
        return self.__pare_list

    @pare_list.setter
    def pare_list(self, pare_list: list) -> None:
        self.__pare_list = pare_list
