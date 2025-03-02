#!/usr/bin/env python3

"""
Base classes for face detection and associated data structures.
"""

class Face:
    """Data on facial features: face, eyes, nose, mouth."""
    def __init__(self):
        self.face_rect = None
        self.left_eye_rect = None
        self.right_eye_rect = None
        self.nose_rect = None
        self.mouth_rect = None

class BaseFaceDetector:
    """Base class for all face detectors."""
    
    def __init__(self, **kwargs):
        """Initialize the detector with common parameters."""
        self.min_face_size = kwargs.get('min_face_size', (30, 30))
        
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face rectangles in (x, y, w, h) format
        """
        raise NotImplementedError("Subclasses must implement detect_faces()")
