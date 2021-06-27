# MISCELLANEOUS FUNCTIONS
import numpy as np


# A function that finds the area of a circle to the left of a line crossing that circle.
# Note that the area to the right of this line is equal to pi*r^2 - the area left of the line.
def find_area_segment(b, t, particle, height, side):
    p1 = np.array([b, 0])
    p2 = np.array([t, height])
    p3 = np.array([particle.x, particle.y])
    d = ((p2[0] - p3[0]) * (p2[1] - p1[1]) + (p2[1] - p3[1]) * (p1[0] - p2[0])) / (
        np.sqrt((p2[1] - p1[1]) ** 2 + (p1[0] - p2[0]) ** 2))
    r = particle.r
    h = r - d
    if d > r:
        area = np.pi * r ** 2
    elif d < -r:
        area = 0
    else:
        area = np.pi * r ** 2 - (r ** 2 * np.arccos((r - h) / r) - (r - h) * np.sqrt(2 * r * h - h ** 2))
    if side == "left":
        return area
    else:
        return np.pi * r ** 2 - area
    return area


# Returns true when one or more points are within the circle described by x, y and radius.
def point_in_circle(center, radius, interest_points):
    for point in interest_points:
        l = np.sqrt((center.x - point.x) ** 2 + (center.y - point.y) ** 2)
        if l < radius:
            return True
    return False


def distance_points(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Checks if a point resides in any of the passed through particles
def in_particle(particle_list, point):
    for particle in particle_list:
        if distance_points(particle, point) < particle.r:
            return True
    return False


# A function to calculate the surface spanned between 2 vectors (L2_error)
def l2(a, b, c, d, height):
    if a != b:
        u = (int(b > a) - int(a > b)) / (max(a, b) - min(a, b))
    else:
        u = 10000000000000
    if c != d:
        v = (int(d > c) - int(c > d)) / (max(c, d) - min(c, d))
    else:
        v = 10000000000000
    if u == v:
        return np.abs(c - a)
    y = (u * v * (c - a)) / (v - u)
    f = lambda y: (y ** 2) * ((u - v) / (v * u)) / 2 + (c - a) * y
    if 0 <= y <= height:
        return np.abs(f(y)) + np.abs(f(height) - f(y))
    else:
        return np.abs(f(height))
