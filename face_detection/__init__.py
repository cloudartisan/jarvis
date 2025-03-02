#!/usr/bin/env python3

"""
Face detection module that provides different detectors and utilities
for working with faces in computer vision.
"""

from .base import Face, BaseFaceDetector
from .dnn_detector import DNNFaceDetector
from .haar_detector import HaarFaceDetector
from .face_recognition import FaceRecognizer

# Default implementation for backward compatibility
FaceDetector = DNNFaceDetector
