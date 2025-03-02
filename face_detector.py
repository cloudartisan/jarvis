#!/usr/bin/env python3


import cv2
import rects
import utils
import colours
from face_detection import Face, DNNFaceDetector, HaarFaceDetector


class FaceDetector:
    """Detects faces and facial features (eyes, nose, mouth) in images."""
    def __init__(self, scale_factor=1.1, min_neighbours=5, flags=None):
        # Using improved parameters from face.py
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours
        self._faces = []
        
        # Initialize face detectors
        try:
            # Create DNN detector with lower confidence threshold for better detection
            self._dnn_detector = DNNFaceDetector(min_confidence=0.5)
            self._use_dnn = True
            print("Using DNN face detector for improved accuracy")
        except Exception as e:
            print(f"Could not initialize DNN face detector: {e}")
            print("Falling back to Haar cascade detectors")
            self._use_dnn = False
        
        # Keep Haar cascade detectors as fallback
        self._face_classifier_alt = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._face_classifier_default = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_default.xml')
        
        self._eye_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._nose_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouth_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """The detected facial features."""
        return self._faces

    def _detect_faces_with_haar(self, gray_image):
        """Detect faces using Haar cascade classifiers with strict parameters."""
        # Set minimum sizes for detection
        min_size = utils.width_height_divided_by(gray_image, 8)
        
        # Use extremely strict parameters to avoid false positives with glasses
        face_rects_default = self._face_classifier_default.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=10,  # Very high to avoid false positives
            minSize=(100, 100)  # Much larger minimum size to avoid detecting eye regions
        )
        
        # Use similarly strict parameters with the alt classifier
        face_rects_alt = self._face_classifier_alt.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=8,  # Very high
            minSize=(100, 100)  # Much larger minimum size
        )
        
        # Combine the results 
        all_face_rects = []
        
        # Add faces from both detectors
        for face_rect in face_rects_default:
            all_face_rects.append(face_rect)
        
        for face_rect in face_rects_alt:
            # Check if this face is already detected (avoid duplicates)
            is_duplicate = False
            for existing_face in all_face_rects:
                # If centers are close, consider it a duplicate
                x1, y1, w1, h1 = existing_face
                x2, y2, w2, h2 = face_rect
                
                # Calculate centers
                c1x, c1y = x1 + w1//2, y1 + h1//2
                c2x, c2y = x2 + w2//2, y2 + h2//2
                
                # Calculate distance between centers
                distance = ((c1x - c2x)**2 + (c1y - c2y)**2)**0.5
                
                # If centers are close, it's the same face
                if distance < (w1 + w2) // 4:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                all_face_rects.append(face_rect)
                
        # Check aspect ratio - faces should be roughly square 
        # This helps filter out false positives like eye regions
        valid_face_rects = []
        for face_rect in all_face_rects:
            x, y, w, h = face_rect
            
            # Calculate aspect ratio
            ratio = float(w) / float(h)
            
            # Valid faces have aspect ratio close to 1 (square)
            if 0.7 <= ratio <= 1.3:
                valid_face_rects.append(face_rect)
        
        return valid_face_rects
        
    def update(self, image):
        """Update the tracked facial features."""
        self._faces = []

        # Prepare the image for detection
        if utils.is_gray(image):
            gray = cv2.equalizeHist(image)
            # We need a color version for DNN-based detection
            color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image  # Keep original for DNN
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)

        # Try DNN detector first if available (more accurate)
        if hasattr(self, '_use_dnn') and self._use_dnn:
            try:
                face_rects = self._dnn_detector.detect_faces(color_image)
                # If no faces found with DNN, fall back to Haar cascades
                if len(face_rects) == 0:
                    face_rects = self._detect_faces_with_haar(gray)
            except Exception as e:
                print(f"DNN face detection failed: {e}")
                face_rects = self._detect_faces_with_haar(gray)
        else:
            # Use Haar cascade detection if DNN not available
            face_rects = self._detect_faces_with_haar(gray)

        # Process detected faces
        if len(face_rects) > 0:
            for face_rect in face_rects:
                face = Face()
                face.face_rect = face_rect

                x, y, w, h = face_rect

                # Seek an eye in the upper-left part of the face.
                search_rect = (x+int(w/7), y, int(w*2/7), int(h/2))
                face.left_eye_rect = self._detect_one_object(
                    self._eye_classifier, gray, search_rect, 64)

                # Seek an eye in the upper-right part of the face.
                search_rect = (x+int(w*4/7), y, int(w*2/7), int(h/2))
                face.right_eye_rect = self._detect_one_object(
                    self._eye_classifier, gray, search_rect, 64)

                # Seek a nose in the middle part of the face.
                search_rect = (x+int(w/4), y+int(h/4), int(w/2), int(h/2))
                face.nose_rect = self._detect_one_object(
                    self._nose_classifier, gray, search_rect, 32)

                # Seek a mouth in the lower-middle part of the face.
                search_rect = (x+int(w/6), y+int(h*2/3), int(w*2/3), int(h/3))
                face.mouth_rect = self._detect_one_object(
                    self._mouth_classifier, gray, search_rect, 16)

                self._faces.append(face)

    def _detect_one_object(
            self, classifier, image, rect, image_size_to_min_size_ratio):
        x, y, w, h = rect
        min_size = utils.width_height_divided_by(
            image, image_size_to_min_size_ratio)

        sub_image = image[y:y+h, x:x+w]

        sub_rects = classifier.detectMultiScale(
            sub_image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=min_size)

        if len(sub_rects) == 0:
            return None

        sub_x, sub_y, sub_w, sub_h = sub_rects[0]
        return (x+sub_x, y+sub_y, sub_w, sub_h)

# Note: Debug drawing methods have been removed as they've been moved to qt_managers.py
# This avoids duplicating code and separates detection logic from visualization