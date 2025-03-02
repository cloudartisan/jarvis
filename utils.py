#!/usr/bin/env python3


import cv2
try:
    import tkinter
except ImportError:
    # If tkinter is not available, provide a fallback
    tkinter = None
import numpy
import scipy.interpolate


def get_screen_resolution():
    if tkinter is None:
        # Fallback values if tkinter is not available
        print("Tkinter not available, using default resolution")
        return 1280, 720
    else:
        root = tkinter.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        print("Screen resolution: {} x {}".format(width, height))
        return width, height


def draw_rectangle(image, rectangle_coordinates):
    """
    Draw a rectangle on the given image using the supplied coordinates.
    """
    (x, y, w, h) = rectangle_coordinates
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def scale_coordinates(coordinates, scale):
    """
    Scale the coordinates by a given scale multiplier. Note, coordinates are
    always integer, so that they can be used with images.
    """
    return tuple(int(coord * scale) for coord in coordinates)


def draw_text(image, text, x, y):
    """
    Draw test on the given image, starting at the supplied x and
    y coordinates.
    """
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def create_flat_view(array):
    """Return a 1D view of an array of any dimensionality."""
    flat_view = array.view()
    flat_view.shape = array.size
    return flat_view


def create_lookup_array(func, length=256):
    """Return a lookup for whole-number inputs to a function.

    The lookup values are clamped to [0, length - 1].
    """
    if func is None:
        return None
    lookup_array = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookup_array[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookup_array


def apply_lookup_array(lookup_array, src, dst):
    """Map a source to a destination using a lookup."""
    if lookup_array is None:
        return
    dst[:] = lookup_array[src]


def create_curve_func(points):
    """Return a function derived from control points."""
    if points is None:
        return None
    num_points = len(points)
    if num_points < 2:
        return None
    xs, ys = zip(*points)
    if num_points < 4:
        kind = 'linear'
        # 'quadratic' is not implemented.
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind,
                                  bounds_error = False)


def create_composite_func(func0, func1):
    """Return a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


def is_gray(image):
    """Return True if the image has one channel per pixel."""
    return image.ndim < 3


def width_height_divided_by(image, divisor):
    """Return an image's dimensions, divided by a value."""
    h, w = image.shape[:2]
    return (int(w/divisor), int(h/divisor))


if __name__ == '__main__':
    get_screen_resolution()