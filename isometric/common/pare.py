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
