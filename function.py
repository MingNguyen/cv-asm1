import cv2
import time
import numpy as np
import math
def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines
def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n
def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []
    
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            if x1 == x2: return None
            slope = (y1-y2) / (x1-x2+1e-5)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)
    
    # Calculate average intersection at lower_border.
    if avg_slope == 0: return None
    x_lane_lower_point = int((lower_border - avg_consts) / (avg_slope+1e-5))
    
    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / (avg_slope+1e-5))
    
    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]

def distance(x1, y1, x2, y2, px, py):
    # Calculate the distance between the point and the vector
    dx = x2 - x1
    dy = y2 - y1
    numerator = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt(dx ** 2 + dy ** 2)
    if denominator == 0: distance = -1
    distance = numerator / denominator

    # Calculate the projection of the point onto the vector
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    proj_x = int(x1 + t * (x2 - x1))
    proj_y = int(y1 + t * (y2 - y1))

    return distance, (proj_x, proj_y)
