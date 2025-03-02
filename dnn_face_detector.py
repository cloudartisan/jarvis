#!/usr/bin/env python3

import cv2
import numpy as np
import os

class DNNFaceDetector:
    """
    A modern face detector using OpenCV's DNN module with a pre-trained model.
    This provides significantly better detection accuracy than Haar cascades,
    particularly with glasses, different poses and lighting conditions.
    """
    
    def __init__(self, min_confidence=0.5):
        """
        Initialize the DNN face detector.
        
        Args:
            min_confidence: Minimum probability to filter weak detections
        """
        self.min_confidence = min_confidence
        
        # Check if we have the required model files
        model_file = "models/opencv_face_detector_uint8.pb"
        config_file = "models/opencv_face_detector.pbtxt"
        
        if not os.path.exists("models"):
            os.makedirs("models")
        
        # Download model files if they don't exist
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            print("Downloading DNN face detection model...")
            self._download_model_files(model_file, config_file)
            
        # Load the DNN face detector
        self.detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    
    def _download_model_files(self, model_file, config_file):
        """Download the required model files if they don't exist"""
        import urllib.request
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # URLs for the model files - updated to working links
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb"
        config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
        
        print(f"Downloading model file from {model_url}")
        urllib.request.urlretrieve(model_url, model_file)
        
        print(f"Downloading config file from {config_url}")
        urllib.request.urlretrieve(config_url, config_file)
        
        print("Download complete!")
    
    def detect_faces(self, image):
        """
        Detect faces in an image using the DNN model.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face rectangles in (x, y, w, h) format
        """
        # Get image dimensions
        (h, w) = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), 
            [104, 117, 123], False, False
        )
        
        # Set the input to the detector
        self.detector.setInput(blob)
        
        # Perform inference
        detections = self.detector.forward()
        
        # Initialize the list of face rectangles
        face_rects = []
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.min_confidence:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                
                # Convert to (x, y, w, h) format
                x = start_x
                y = start_y
                w = end_x - start_x
                h = end_y - start_y
                
                # Add to results if valid dimensions (avoid negative values)
                if w > 0 and h > 0:
                    face_rects.append((x, y, w, h))
        
        return face_rects