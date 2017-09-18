#!/usr/bin/env python


import cv2
import numpy
import utils


def recolour_RC(src, dst):
    """Simulate conversion from BGR to RC (red, cyan).

    The source and destination images must both be in BGR format.

    Blues and greens are replaced with cyans. The effect is similar
    to Technicolor Process 2 (used in early color movies) and CGA
    Palette 3 (used in early color PCs).

    Pseudocode:
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)


def recolour_RGV(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).

    The source and destination images must both be in BGR format.

    Blues are desaturated. The effect is similar to Technicolor
    Process 1 (used in early color movies).

    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolour_CMV(src, dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).

    The source and destination images must both be in BGR format.

    Yellows are desaturated. The effect is similar to CGA Palette 1
    (used in early color PCs).

    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


def blend(foreground_srv, background_src, dst, alpha_mask):
    # Calculate the normalised alpha mask.
    max_alpha = numpy.iinfo(alpha_mask.dtype).max
    normalised_alpha_mask = (1.0 / max_alpha) * alpha_mask

    # Calculate the normalised inverse alpha mask.
    normalised_inverse_alpha_mask = \
        numpy.ones_like(normalised_alpha_mask)
    normalised_inverse_alpha_mask[:] = \
        normalised_inverse_alpha_mask - normalised_alpha_mask

    # Split the channels from the sources.
    foreground_channels = cv2.split(foreground_srv)
    background_channels = cv2.split(background_src)

    # Blend each channel.
    num_channels = len(foreground_channels)
    i = 0
    while i < num_channels:
        background_channels[i][:] = \
            normalised_alpha_mask * foreground_channels[i] + \
            normalised_inverse_alpha_mask * background_channels[i]
        i += 1

    # Merge the blended channels into the destination.
    cv2.merge(background_channels, dst)


def stroke_edges(src, dst, blur_k_size=7, edge_k_size=5):
    if blur_k_size >= 3:
        blurred_src = cv2.medianBlur(src, blur_k_size)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize = edge_k_size)
    normalised_inverse_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalised_inverse_alpha
    cv2.merge(channels, dst)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""
    def __init__(self, v_func=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._v_lookup_array = utils.create_lookup_array(v_func, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        src_flat_view = utils.flat_view(src)
        dst_flat_view = utils.flat_view(dst)
        utils.apply_lookup_array(
            self._v_lookup_array, src_flat_view, dst_flat_view)


class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""
    def __init__(self, v_points, dtype=numpy.uint8):
        VFuncFilter.__init__(
            self, utils.create_curve_func(v_points), dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""
    def __init__(
            self, v_func=None, b_func=None, g_func=None,
            r_func=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._b_lookup_array = utils.create_lookup_array(
            utils.create_composite_func(b_func, v_func), length)
        self._g_lookup_array = utils.create_lookup_array(
            utils.create_composite_func(g_func, v_func), length)
        self._r_lookup_array = utils.create_lookup_array(
            utils.create_composite_func(r_func, v_func), length)

    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.apply_lookup_array(self._b_lookup_array, b, b)
        utils.apply_lookup_array(self._g_lookup_array, g, g)
        utils.apply_lookup_array(self._r_lookup_array, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""
    def __init__(
            self, v_points=None, b_points=None,
            g_points=None, r_points=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(
            self,
            utils.create_curve_func(v_points),
            utils.create_curve_func(b_points),
            utils.create_curve_func(g_points),
            utils.create_curve_func(r_points), dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            b_points=[(0,20),(255,235)],
            g_points=[(0,0),(56,39),(208,226),(255,255)],
            r_points=[(0,0),(56,22),(211,255),(255,255)],
            dtype=dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            v_points=[(0,0),(23,20),(157,173),(255,255)],
            b_points=[(0,0),(41,46),(231,228),(255,255)],
            g_points=[(0,0),(52,47),(189,196),(255,255)],
            r_points=[(0,0),(69,69),(213,218),(255,255)],
            dtype = dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            b_points=[(0,0),(35,25),(205,227),(255,255)],
            g_points=[(0,0),(27,21),(196,207),(255,255)],
            r_points=[(0,0),(59,54),(202,210),(255,255)],
            dtype=dtype)


class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            v_points=[(0,0),(128,118),(221,215),(255,255)],
            b_points=[(0,0),(25,21),(122,153),(165,206),(255,255)],
            g_points=[(0,0),(25,21),(95,102),(181,208),(255,255)],
            r_points=[(0,0),(41,28),(183,209),(255,255)],
            dtype=dtype)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""
    def __init__(self):
        kernel = numpy.array(
            [[0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array(
            [[-2, -1, 0],
             [-1,  1, 1],
             [ 0,  1, 2]])
        VConvolutionFilter.__init__(self, kernel)
