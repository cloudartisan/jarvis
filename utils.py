#!/usr/bin/env python


import cv2
import Tkinter


def get_screen_resolution():
    root = Tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    print "Screen resolution: {} x {}".format(width, height)
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


if __name__ == '__main__':
    get_screen_resolution()
