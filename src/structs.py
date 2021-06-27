from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


class Particle(NamedTuple):
    x: float
    y: float
    r: float


class Distance(NamedTuple):
    x: float
    y: float
    d: float


class Section:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.points = []

    def add_point(self, point):
        self.points += [point]
