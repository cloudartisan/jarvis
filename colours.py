#!/usr/bin/env python3
"""
Color constants and utilities for computer vision applications.
BGR is the default format used by OpenCV.
"""

# BGR colour constants
WHITE_BGR = (255, 255, 255)
BLACK_BGR = (0, 0, 0)
RED_BGR = (0, 0, 255)
YELLOW_BGR = (0, 255, 255)
GREEN_BGR = (0, 255, 0)
BLUE_BGR = (255, 0, 0)
CYAN_BGR = (255, 255, 0)
MAGENTA_BGR = (255, 0, 255)
ORANGE_BGR = (0, 165, 255)
PURPLE_BGR = (128, 0, 128)
PINK_BGR = (203, 192, 255)
GRAY_BGR = (128, 128, 128)

# Face detection colours
PRIMARY_COLOUR = (180, 180, 255)
SECONDARY_COLOUR = (128, 200, 128)

# Face feature colours
FACE_COLOUR = (180, 180, 255)
LEFT_EYE_COLOUR = (200, 162, 124)
RIGHT_EYE_COLOUR = (200, 162, 124)
NOSE_COLOUR = (128, 200, 128)
MOUTH_COLOUR = (128, 200, 200)

# Text colours
TEXT_OUTLINE_COLOUR = (0, 0, 0)
TEXT_COLOUR = (255, 255, 255)
HIGHLIGHT_TEXT_COLOUR = (255, 255, 255)

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
