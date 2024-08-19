import ezdxf
from math import cos, sin, pi, sqrt
import numpy as np

def extract_lines_from_dxf(file_path):
    evaluate_lines = []
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = msp.query('LINE')
    for line in lines:
        evaluate_lines.append(((line.dxf.start[0], line.dxf.start[1]), (line.dxf.end[0], line.dxf.end[1])))
    return evaluate_lines

def distance(point1, point2):
    """2つの点間のユークリッド距離を計算します"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_line_near(line1, line2, threshold):
    """2つの線が指定されたしきい値以下の距離で近接しているかを判定します"""
    start_distance = distance(line1[0], line2[0])
    end_distance = distance(line1[1], line2[1])
    return start_distance <= threshold and end_distance <= threshold

def compute_metrics(predicted, ground_truth):
    threshold = 10
    tp = 0
    matched = set()

    for pred_line in predicted:
        for gt_line in ground_truth:
            if is_line_near(pred_line, gt_line, threshold) and gt_line not in matched:
                tp += 1
                matched.add(gt_line)
                break

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return {
        "error": fp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class GenDxf:
    """Generate dxf file"""
    def __init__(self, _cfg) -> None:
        self.cfg = _cfg
        self.__doc = ezdxf.new()
        dimstyle = self.__doc.dimstyles.new('custom_dimstyle')
        dimstyle.dxf.dimtxt = self.cfg['isometric']['dimtxt']
        dimstyle.dxf.dimdec = self.cfg['isometric']['dimdex']
        dimstyle.dxf.dimasz = self.cfg['isometric']['dimasz']
        dimstyle.dxf.dimblk = self.cfg['isometric']['dimblk']
        dimstyle.dxf.dimclrd = self.cfg['isometric']['dimclrd']
        dimstyle.dxf.dimclre = self.cfg['isometric']['dimclre']
        self.__msp = self.__doc.modelspace()
        self.__position = (self.cfg['isometric']['initial_position'],
                           self.cfg['isometric']['initial_position'])
        self.__pre_position = (self.cfg['isometric']['initial_position'],
                               self.cfg['isometric']['initial_position'])

        self.__add_degree: float = 22.5
        self.__yaw_up_rem: float = 0
        self.__yaw_down_rem: float = 0
        self.__epsilon: float = 1e-10

        self.__evaluate_lines: list = []
        self.__pre_distance: float = 0
        self.__is_direction: bool = False

    def _draw_forward(self, point1: tuple, distance: float, direction: bool):
        direction_radian = pi/6
        if direction == 0:
            direction_radian = -pi/6
        elif direction == 1:
            direction_radian = pi/6
        elif direction == 2:
            direction_radian = 5*pi/6
        elif direction == 3:
            direction_radian = -5*pi/6
        point2 = (int(distance*cos(direction_radian)+point1[0]), int(distance*sin(direction_radian)+point1[1]))
        self.__msp.add_line(point1, point2)
        distance = int(sqrt((point2[1] - point1[1]) * (point2[1] - point1[1]) + (point2[0] - point1[0]) * (point2[0] - point1[0])))
        if point1[0] < point2[0]:
            self.__msp.add_aligned_dim(
                p1=point1,
                p2=point2,
                distance=-self.cfg['isometric']['dimdistance'],
                dimstyle="custom_dimstyle",
                text=str(distance)
                ).render()
        else:
            self.__msp.add_aligned_dim(
                p1=point2,
                p2=point1,
                distance=-self.cfg['isometric']['dimdistance'],
                dimstyle="custom_dimstyle",
                text=str(distance)
                ).render()
        self.__evaluate_lines.append((point1, point2))
        return point2

    def _draw_forward_only(self, point: tuple, distance: float, direction: bool) -> None:
        direction_radian = pi/6
        if direction == 0:
            direction_radian *= -1
        self.__msp.add_line(point, (distance*cos(direction_radian) + point[0], distance*sin(direction_radian) + point[1]))
    
    def _draw_downward(self, point1, distance):
        point2 = (point1[0], point1[1] - distance)
        self.__msp.add_line(point1, point2)
        distance = int(sqrt((point2[1] - point1[1]) * (point2[1] - point1[1]) + (point2[0] - point1[0]) * (point2[0] - point1[0])))
        self.__msp.add_aligned_dim(
            p1=point1,
            p2=point2,
            distance=-self.cfg['isometric']['dimdistance'],
            dimstyle="custom_dimstyle",
            text=str(distance)
            ).render()
        self.__evaluate_lines.append((point1, point2))
        return point2

    def _draw_downward_only(self, point, distance) -> None:
        self.__msp.add_line(point, (point[0], point[1] - distance))

    def _draw_upward(self, point1, distance):
        point2 = (point1[0], point1[1] + distance)
        self.__msp.add_line(point1, point2)
        self.__msp.add_linear_dim(
            base=(point1[0] + self.cfg['isometric']['dimdistance'], (point1[1] + point1[1] - distance) / 2),
            p1=point1,
            p2=point2,
            angle=90,
            dimstyle="custom_dimstyle",
        ).render()
        self.__evaluate_lines.append((point1, point2))
        return point2

    def _draw_upward_only(self, point, distance) -> None:
        self.__msp.add_line(point, (point[0], point[1] + distance))
    
    def _degree_normalize(self, _yaw: float):
        if _yaw > 0:
            return _yaw, False
        else:
            return _yaw + 180, True

    def _judge_direction(self, _yaw: float):
        yaw, is_direction = self._degree_normalize(_yaw)
        direction = int(yaw < self.__yaw_up_rem and yaw > self.__yaw_down_rem)
        if is_direction != self.__is_direction:
            direction += 2
        return direction

    def isometric(self, isometric_info):
        """generate isometric dxf"""
        connect_count = -1
        self.__evaluate_lines = []

        for result in isometric_info:
            print(result.position1, result.position2, result.relationship, result.name1, result.name2, result.distance)
            if result.distance < self.__epsilon:
                continue
            if connect_count == -1:
                yaw, self.__is_direction = self._degree_normalize(result.yaw)
                self.__yaw_up_rem = yaw + self.__add_degree
                self.__yaw_down_rem = yaw - self.__add_degree
            connect_count += 1
            if result.relationship == 'forward':
                direction = self._judge_direction(result.yaw)
                if result.name2 != 'None':
                    self.__position = self._draw_forward(point1=self.__position, distance=result.distance, direction=direction)
                else:
                    self._draw_forward_only(point=self.__pre_position, distance=self.__pre_distance, direction=direction)
            elif result.relationship == 'downward':
                if result.name2 != 'None':
                    self.__position = self._draw_downward(self.__position, result.distance)
                else:
                    self._draw_downward_only(self.__pre_position, self.__pre_distance)
            elif result.relationship == 'upward':
                if result.name2 != 'None':
                    self.__position = self._draw_upward(self.__position, result.distance)
                else:
                    self._draw_upward_only(self.__pre_position, self.__pre_distance)
            print(self.__pre_distance, connect_count, result.name2, connect_count == 0, result.name2 != 'None')
            if connect_count == 0 and result.name2 != 'None':
                self.__pre_distance = result.distance
            if (result.name1 == 'elbow' and connect_count == 1) or (result.name1 == 'tee' and connect_count == 2):
                self.__pre_position = self.__position
                connect_count = 0
        self.__doc.saveas(self.cfg['isometric']['output_dxf_path'])
        file = "./data/isometric/labels/Drawing2.dxf"

        coords = extract_lines_from_dxf(file)
        # print(self.__evaluate_lines)
        # print(coords)
        # print(compute_metrics(self.__evaluate_lines, coords))
