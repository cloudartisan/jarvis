#!/usr/bin/env python3
"""
Color constants and utilities for computer vision applications.
BGR is the default format used by OpenCV.
"""

# BGR color constants (BGR order as used by OpenCV)
WHITE_BGR = (255, 255, 255)
BLACK_BGR = (0, 0, 0)
RED_BGR = (0, 0, 255)     # Red is (0,0,255) in BGR
YELLOW_BGR = (0, 255, 255)
GREEN_BGR = (0, 255, 0)
BLUE_BGR = (255, 0, 0)    # Blue is (255,0,0) in BGR
CYAN_BGR = (255, 255, 0)
MAGENTA_BGR = (255, 0, 255)
ORANGE_BGR = (0, 165, 255)
PURPLE_BGR = (128, 0, 128)
PINK_BGR = (203, 192, 255)
GRAY_BGR = (128, 128, 128)

# Face feature colors
FACE_COLOR = WHITE_BGR
LEFT_EYE_COLOR = RED_BGR
RIGHT_EYE_COLOR = YELLOW_BGR
NOSE_COLOR = GREEN_BGR
MOUTH_COLOR = BLUE_BGR

# Text colors
TEXT_OUTLINE_COLOR = BLACK_BGR
TEXT_COLOR = WHITE_BGR
HIGHLIGHT_TEXT_COLOR = YELLOW_BGR

# HSV min/max values for the green range
GREEN_HSV_RANGE = {
    'h_min': 42, 'h_max': 92,
    's_min': 62, 's_max': 255,
    'v_min': 63, 'v_max': 235
}

# HSV min/max values for the red range
RED_HSV_RANGE = {
    'h_min': 0, 'h_max': 179,
    's_min': 131, 's_max': 255,
    'v_min': 126, 'v_max': 255
}

# Backward compatibility
GREEN_H_MIN = GREEN_HSV_RANGE['h_min']
GREEN_H_MAX = GREEN_HSV_RANGE['h_max']
GREEN_S_MIN = GREEN_HSV_RANGE['s_min']
GREEN_S_MAX = GREEN_HSV_RANGE['s_max']
GREEN_V_MIN = GREEN_HSV_RANGE['v_min']
GREEN_V_MAX = GREEN_HSV_RANGE['v_max']

RED_H_MIN = RED_HSV_RANGE['h_min']
RED_H_MAX = RED_HSV_RANGE['h_max']
RED_S_MIN = RED_HSV_RANGE['s_min']
RED_S_MAX = RED_HSV_RANGE['s_max']
RED_V_MIN = RED_HSV_RANGE['v_min']
RED_V_MAX = RED_HSV_RANGE['v_max']
