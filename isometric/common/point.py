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