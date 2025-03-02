#\!/usr/bin/env python3

import cv2
import numpy as np
import os
from .base import BaseFaceDetector

class HaarFaceDetector(BaseFaceDetector):
    """
    Face detector using Haar cascade classifiers.
    Simpler but less accurate than DNN-based detection.
    """
    
    def __init__(self, 
                scale_factor=1.1, 
                min_neighbors=5, 
                classifier_file=None,
                **kwargs):
        """
        Initialize the Haar cascade face detector.
        
        Args:
            scale_factor: How much the image size is reduced at each image scale
            min_neighbors: How many neighbors each candidate rectangle should have
            classifier_file: Path to cascade classifier XML file 
        """
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Use the default classifier if none provided
        if classifier_file is None:
            self.classifier_file = 'cascades/haarcascade_frontalface_default.xml'
        else:
            self.classifier_file = classifier_file
            
        # Load the cascade classifier
        self.detector = cv2.CascadeClassifier(self.classifier_file)
    
    def detect_faces(self, image):
        """
        Detect faces in an image using Haar cascades.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face rectangles in (x, y, w, h) format
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply histogram equalization to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )
        
        return faces
