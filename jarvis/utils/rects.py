#!/usr/bin/env python3


import cv2
import numpy


def outline_rect(image, rect, colour):
    """Draw a rectangle around a region of interest in an image."""
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), colour, 1)


def copy_rect(src, dst, src_rect, dst_rect,
              interpolation=cv2.INTER_LINEAR):
    """Copy part of the source image to the destination image."""
    # Validate and normalize rectangles
    x_src, y_src, w_src, h_src = src_rect
    x_dst, y_dst, w_dst, h_dst = dst_rect
    
    # If sizes don't match, resize the source ROI
    if w_src != w_dst or h_src != h_dst:
        src_roi = src[y_src:y_src+h_src, x_src:x_src+w_src]
        dst_roi = cv2.resize(src_roi, (w_dst, h_dst), 
                             interpolation=interpolation)
        dst[y_dst:y_dst+h_dst, x_dst:x_dst+w_dst] = dst_roi
    else:
        # No resizing needed
        dst_roi = dst[y_dst:y_dst+h_dst, x_dst:x_dst+w_dst]
        src_roi = src[y_src:y_src+h_src, x_src:x_src+w_src]
        dst_roi[:] = src_roi[:]


def swap_rects(src, dst, rects, 
               interpolation=cv2.INTER_LINEAR):
    """Copy the source with two ROIs swapped."""
    if len(rects) < 2:
        return
    
    # Copy the whole image
    dst[:] = src[:]
    
    # Swap first two rectangles
    x1, y1, w1, h1 = rects[0]
    x2, y2, w2, h2 = rects[1]
    
    # Create temporary storage for the first ROI
    temp = src[y1:y1+h1, x1:x1+w1].copy()
    
    # Copy the second ROI to the first position
    copy_rect(src, dst, (x2, y2, w2, h2), 
              (x1, y1, w1, h1), interpolation)
    
    # Copy the first ROI to the second position from the temp copy
    dst[y2:y2+h2, x2:x2+w2] = cv2.resize(
        temp, (w2, h2), interpolation=interpolation)