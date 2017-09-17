#!/usr/bin/env python


import sys
import os.path

import cv2

import utils


HAAR_CASCADE_CLASSIFIER = 'config/haarcascade_frontalface_default.xml'


class FaceDetector:
    def __init__(self, classifier_file=HAAR_CASCADE_CLASSIFIER):
        self.classifier_file = classifier_file
        self.face_cascade = cv2.CascadeClassifier(classifier_file)

    def detect(self, image):
        """
        Convert the image to grayscale as OpenCV face detector expects gray
        images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return None, None

        # FIXME assumes there will be only one face
        # Extract the face area
        (x, y, w, h) = faces[0]

        # Return only the face part of the image
        return gray[y:y+w, x:x+h], faces[0]


class FaceRecogniser:
    def __init__(self):
        self.subjects = {}
        self.face_detector = FaceDetector()

    def train(self, training_data_path):
        #cv2.namedWindow('Training...', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('Training...', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_width, screen_height = utils.get_screen_resolution()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        #cv2.namedWindow('Training...', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Training...')
        #cv2.resizeWindow('Training...', window_width, window_height)

        dirs = os.listdir(training_data_path)

        # Examine each directory prefixed with "subject"
        # (e.g., subject1, subject2, ... subjectN)
        for dir_name in dirs:
            if not dir_name.startswith('subject'):
                continue
            subject_id = int(dir_name.replace('subject', ''))
            subject_path = os.path.join(training_data_path, dir_name)
            subject_image_files = os.listdir(subject_path)
            self.subjects[subject_id] = {
                'name' : None,
                'data_path' : subject_path,
                'image_files' : subject_image_files,
                'faces' : [],
            }
            # Read each image, detect the face, add the detected face to the
            # subject's list of faces
            for image_name in subject_image_files:
                # Ignore system files like .DS_Store
                if image_name.startswith('.') or image_name == 'name.txt':
                    continue;
                image_path = os.path.join(subject_path, image_name)
                image = cv2.imread(image_path)

                # Display an image window to show the image
                #cv2.imshow('Training...', image)
                small_image = cv2.resize(image, None, fx = 0.1, fy = 0.1)
                cv2.imshow('Training...', small_image)
                cv2.waitKey(100)

                # Detect and record face
                face, rect = self.face_detector.detect(image)
                if face is not None:
                    small_rectangle_coordinates = utils.scale_coordinates(rect, 0.1)
                    utils.draw_rectangle(small_image, small_rectangle_coordinates)
                    cv2.imshow('Training...', small_image)
                    cv2.waitKey(100)
                    self.subjects[subject_id]['faces'].append(face)

        # Clean up after ourselves
        cv2.destroyWindow('Training...')
        cv2.waitKey(1)
        cv2.destroyWindow('Training...')


def main():
    face_recogniser = FaceRecogniser()
    face_recogniser.train(sys.argv[1])

if __name__ == '__main__':
    main()
