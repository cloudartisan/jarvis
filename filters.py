#!/usr/bin/env python3


import cv2
import numpy
import utils


def stroke_edges(src, dst, blur_k_size = 7, edges_k_size = 5):
    if blur_k_size >= 3:
        blurred_src = cv2.medianBlur(src, blur_k_size)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize = edges_k_size)
    normalized_inverse_alpha = (1.0/255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    cv2.merge(channels, dst)


class VConvolutionFilter:
    def __init__(self, kernel):
        self._kernel = kernel
    
    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]])
        VConvolutionFilter.__init__(self, kernel)


class BGRFuncFilter:
    def __init__(self, v_func = None, b_func = None, g_func = None,
                 r_func = None):
        self._v_func = v_func
        self._b_func = b_func
        self._g_func = g_func
        self._r_func = r_func
        
        self._v_lookup_array = utils.create_lookup_array(v_func)
        self._b_lookup_array = utils.create_lookup_array(b_func)
        self._g_lookup_array = utils.create_lookup_array(g_func)
        self._r_lookup_array = utils.create_lookup_array(r_func)
    
    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        if self._b_lookup_array is not None:
            utils.apply_lookup_array(self._b_lookup_array, b, b)
        if self._g_lookup_array is not None:
            utils.apply_lookup_array(self._g_lookup_array, g, g)
        if self._r_lookup_array is not None:
            utils.apply_lookup_array(self._r_lookup_array, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, v_points = None, b_points = None,
                 g_points = None, r_points = None):
        v_func = utils.create_curve_func(v_points)
        b_func = utils.create_curve_func(b_points)
        g_func = utils.create_curve_func(g_points)
        r_func = utils.create_curve_func(r_points)
        
        BGRFuncFilter.__init__(self, v_func, b_func, g_func, r_func)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    def __init__(self):
        BGRCurveFilter.__init__(
            self,
            # Violet curve.
            [(0, 20), (128, 128), (255, 235)],
            # Blue curve.
            [(0, 0), (64, 50), (128, 128), (192, 150), (255, 255)],
            # Green curve.
            [(0, 50), (128, 128), (255, 255)],
            # Red curve.
            [(0, 50), (192, 192), (255, 255)])


class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self):
        BGRCurveFilter.__init__(
            self,
            # Violet curve.
            [(0, 0), (23, 20), (157, 173), (255, 255)],
            # Blue curve.
            [(0, 0), (41, 46), (231, 228), (255, 255)],
            # Green curve.
            [(0, 0), (52, 47), (189, 196), (255, 255)],
            # Red curve.
            [(0, 0), (69, 69), (213, 218), (255, 255)])


class BGRProviaCurveFilter(BGRCurveFilter):
    def __init__(self):
        BGRCurveFilter.__init__(
            self,
            # Violet curve.
            [(0, 0), (35, 25), (126, 126), (255, 255)],
            # Blue curve.
            [(0, 0), (94, 74), (187, 198), (255, 255)],
            # Green curve.
            [(0, 0), (65, 59), (176, 187), (255, 255)],
            # Red curve.
            [(0, 0), (54, 55), (158, 169), (255, 255)])


class BGRVelviaCurveFilter(BGRCurveFilter):
    def __init__(self):
        BGRCurveFilter.__init__(
            self,
            # Violet curve.
            [(0, 0), (122, 144), (214, 188), (255, 255)],
            # Blue curve.
            [(0, 0), (18, 18), (142, 144), (255, 255)],
            # Green curve.
            [(0, 0), (85, 98), (189, 216), (255, 255)],
            # Red curve.
            [(0, 0), (86, 73), (175, 180), (255, 255)])