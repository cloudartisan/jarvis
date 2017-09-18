import cv2
import sys


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while cv2.waitKey(1) & 0xFF != ord('q'):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Annotate the frame with the number of faces found
    found = "Faces: {}".format(len(faces))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, found, (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Type q to quit", (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
