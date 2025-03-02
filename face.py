#!/usr/bin/env python3


import sys
import os.path

import cv2
import numpy

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
        self.face_detector = FaceDetector()
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create()
        self.subject_to_label = {}

    def train(self, training_data_path):
        faces = []
        labels = []
        cv2.namedWindow('Training...')
        cv2.moveWindow('Training...', 0, 0)

        dirs = os.listdir(training_data_path)
        print(dirs)

        label = 0
        for dir_name in dirs:
            if dir_name.startswith('.'):
                continue
            subject = dir_name
            print(subject, label)
            subject_path = os.path.join(training_data_path, dir_name)
            subject_image_files = os.listdir(subject_path)
            # Read each image, detect the face, add the detected face to the
            # subject's list of faces
            for image_name in subject_image_files:
                # Ignore system files like .DS_Store
                if image_name.startswith('.') or image_name == 'name.txt':
                    continue
                print(image_name)
                image_path = os.path.join(subject_path, image_name)
                image = cv2.imread(image_path)

                # Display an image window to show the image
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
                    faces.append(face)
                    labels.append(label)

            label += 1

        # Clean up after ourselves
        cv2.destroyWindow('Training...')
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        self.face_recogniser.train(faces, numpy.array(labels))


def main():
    face_recogniser = FaceRecogniser()
    face_recogniser.train(sys.argv[1])

if __name__ == '__main__':
    main()