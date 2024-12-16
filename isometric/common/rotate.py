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
