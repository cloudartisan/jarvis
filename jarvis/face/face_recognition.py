#!/usr/bin/env python3

import os
import cv2
import numpy as np
from jarvis.utils import helpers as utils
from .haar_detector import HaarFaceDetector

class FaceRecognizer:
    """
    Recognizes faces based on trained data using Local Binary Pattern Histograms.
    """
    def __init__(self, face_detector=None):
        """
        Initialize the face recognizer.
        
        Args:
            face_detector: A face detector instance (defaults to HaarFaceDetector)
        """
        if face_detector is None:
            self.face_detector = HaarFaceDetector()
        else:
            self.face_detector = face_detector
            
        # Create the recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.subject_to_label = {}
        self.label_to_subject = {}
        self.is_trained = False
    
    def train(self, training_data_path, show_progress=True):
        """
        Train the recognizer with images from the training data path.
        Each subdirectory should be named after the subject (person) it contains.
        
        Args:
            training_data_path: Path to directory containing subject subdirectories
            show_progress: Whether to show visual progress during training
        """
        faces = []
        labels = []
        
        if show_progress:
            cv2.namedWindow('Training...')
            cv2.moveWindow('Training...', 0, 0)

        dirs = os.listdir(training_data_path)
        print(f"Found {len(dirs)} subjects")

        label = 0
        # Process each subject directory
        for dir_name in dirs:
            if dir_name.startswith('.'):
                continue
                
            subject = dir_name
            print(f"Processing subject: {subject} (label {label})")
            
            # Map the subject name to a numeric label
            self.subject_to_label[subject] = label
            self.label_to_subject[label] = subject
            
            subject_path = os.path.join(training_data_path, dir_name)
            subject_image_files = os.listdir(subject_path)
            
            # Read each image, detect the face, add the detected face to the
            # subject's list of faces
            for image_name in subject_image_files:
                # Ignore system files like .DS_Store
                if image_name.startswith('.') or image_name == 'name.txt':
                    continue
                    
                print(f"  Processing image: {image_name}")
                image_path = os.path.join(subject_path, image_name)
                image = cv2.imread(image_path)

                if show_progress:
                    # Display an image window to show the image
                    small_image = cv2.resize(image, None, fx=0.1, fy=0.1)
                    cv2.imshow('Training...', small_image)
                    cv2.waitKey(100)

                # Detect faces
                face_rects = self.face_detector.detect_faces(image)
                
                if len(face_rects) > 0:
                    # Use the first detected face
                    x, y, w, h = face_rects[0]
                    face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
                    
                    if show_progress:
                        small_rect = utils.scale_coordinates((x, y, w, h), 0.1)
                        utils.draw_rectangle(small_image, small_rect)
                        cv2.imshow('Training...', small_image)
                        cv2.waitKey(100)
                        
                    faces.append(face)
                    labels.append(label)

            label += 1

        # Clean up after ourselves
        if show_progress:
            cv2.destroyWindow('Training...')
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # Train the recognizer if we have faces
        if len(faces) > 0:
            print(f"Training with {len(faces)} faces")
            self.face_recognizer.train(faces, np.array(labels))
            self.is_trained = True
            print("Training complete!")
        else:
            print("No faces found for training")
    
    def recognize(self, image):
        """
        Recognize a face in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (subject_name, confidence) or (None, None) if no face detected
        """
        if not self.is_trained:
            print("Recognizer not trained")
            return None, None
            
        # Detect faces
        face_rects = self.face_detector.detect_faces(image)
        
        if len(face_rects) == 0:
            return None, None
            
        # Use the first detected face
        x, y, w, h = face_rects[0]
        face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        
        # Recognize the face
        label, confidence = self.face_recognizer.predict(face)
        
        # Return the subject name
        if label in self.label_to_subject:
            return self.label_to_subject[label], confidence
        else:
            return None, confidence
