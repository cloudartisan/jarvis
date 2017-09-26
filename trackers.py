#!/usr/bin/env python


import cv2
import rects
import utils


class Face(object):
    """Data on facial features: face, eyes, nose, mouth."""
    def __init__(self):
        self.face_rect = None
        self.left_eye_rect = None
        self.right_eye_rect = None
        self.nose_rect = None
        self.mouth_rect = None


class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth."""
    def __init__(self, scale_factor=1.2, min_neighbours=2, flags=None):
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours
        self._faces = []

        self._face_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._eye_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._nose_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouth_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces

    def update(self, image):
        """Update the tracked facial features."""
        self._faces = []

        if utils.is_gray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)

        min_size = utils.width_height_divided_by(image, 8)

        face_rects = self._face_classifier.detectMultiScale(
            image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=min_size
        )

        if face_rects is not None:
            for face_rect in face_rects:
                face = Face()
                face.face_rect = face_rect

                x, y, w, h = face_rect

                # Seek an eye in the upper-left part of the face.
                search_rect = (x+w/7, y, w*2/7, h/2)
                face.left_eye_rect = self._detect_one_object(
                    self._eye_classifier, image, search_rect, 64)

                # Seek an eye in the upper-right part of the face.
                search_rect = (x+w*4/7, y, w*2/7, h/2)
                face.right_eye_rect = self._detect_one_object(
                    self._eye_classifier, image, search_rect, 64)

                # Seek a nose in the middle part of the face.
                search_rect = (x+w/4, y+h/4, w/2, h/2)
                face.nose_rect = self._detect_one_object(
                    self._nose_classifier, image, search_rect, 32)

                # Seek a mouth in the lower-middle part of the face.
                search_rect = (x+w/6, y+h*2/3, w*2/3, h/3)
                face.mouth_rect = self._detect_one_object(
                    self._mouth_classifier, image, search_rect, 16)

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

    def draw_debug_text(self, image):
        """Draw text indicating the number of detected faces."""
        if utils.is_gray(image):
            text_colour = 255
        else:
            text_colour = (255, 255, 255)

        found = "Faces: {}".format(len(self.faces))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, found, (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def draw_debug_rects(self, image):
        """Draw rectangles around the tracked facial features."""
        if utils.is_gray(image):
            face_colour = 255
            left_eye_colour = 255
            right_eye_colour = 255
            nose_colour = 255
            mouth_colour = 255
        else:
            face_colour = (255, 255, 255) # white
            left_eye_colour = (0, 0, 255) # red
            right_eye_colour = (0, 255, 255) # yellow
            nose_colour = (0, 255, 0) # green
            mouth_colour = (255, 0, 0) # blue

        for face in self.faces:
            rects.outline_rect(image, face.face_rect, face_colour)
            rects.outline_rect(image, face.left_eye_rect, left_eye_colour)
            rects.outline_rect(image, face.right_eye_rect, right_eye_colour)
            rects.outline_rect(image, face.nose_rect, nose_colour)
            rects.outline_rect(image, face.mouth_rect, mouth_colour)
