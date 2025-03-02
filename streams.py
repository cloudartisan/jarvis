#!/usr/bin/env python3


import logging
import time
from io import BytesIO
from threading import Thread, Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy
from PIL import Image


class DummyStream:
    def __init__(self):
        self.stopped = False
        self._frame = None

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    def read(self):
        return self.frame

    def start(self):
        return self

    def stop(self):
        self.stopped = True


class WebcamVideoStream:
    def __init__(self, device=0, should_mirror=False):
        self.should_mirror = should_mirror
        self.stopped = False
        self.grabbed = False
        self._frame = None
        self._stream = cv2.VideoCapture(device)

    def read(self):
        if self.should_mirror and self._frame is not None:
            return numpy.fliplr(self._frame).copy()
        else:
            return self._frame
            
    def get(self, propId):
        """Get a property from the underlying VideoCapture object."""
        return self._stream.get(propId)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self._frame = self._stream.read()

    def stop(self):
        self.stopped = True


class WebRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while not self.server.stopped:
                frame = self.server.camera_feed.read()
                if frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                s = BytesIO()
                image.save(s, 'JPEG')
                self.wfile.write(b'--jpgboundary')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(s.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="/stream.mjpg"/>')
            self.wfile.write(b'</body></html>')


class ThreadedWebStream(Thread):
    def __init__(self, camera_feed, ip='127.0.0.1', port=8000):
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass
        super(ThreadedWebStream, self).__init__()
        self.ip = ip
        self.port = port
        self.server = ThreadedHTTPServer((ip, port), WebRequestHandler)
        self.server.camera_feed = camera_feed
        self.server.stopped = True

    def run(self):
        self.server.stopped = False
        self.server.serve_forever()

    def stop(self):
        self.server.stopped = True
        self.server.shutdown()

    def __str__(self):
        return "{}:{}".format(self.ip, self.port)


class MediaRecorder(Thread):
    """
    Handles capturing frames from a video stream and recording them as screenshots or video files.
    Runs in a separate thread to avoid blocking the main application.
    """
    def __init__(self, capture, should_mirror_preview=False):
        """
        Initialize the media recorder.
        
        Args:
            capture: A video capture source with a read() method that returns frames
            should_mirror_preview: Whether to mirror the frames when displaying
        """
        super(MediaRecorder, self).__init__()
        self.daemon = True
        self.should_mirror_preview = should_mirror_preview
        self.stopped = False
        
        self._capture = capture
        self._frame = None
        self._frame_lock = Lock()
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None
    
    def run(self):
        """Main thread function that captures frames and handles recording."""
        while not self.stopped:
            frame = self._capture.read()
            if frame is None:
                continue

            # Update the FPS estimate
            if self._frames_elapsed == 0:
                self._start_time = time.time()
            else:
                time_elapsed = time.time() - self._start_time
                self._fps_estimate = self._frames_elapsed / time_elapsed
            self._frames_elapsed += 1

            # Write to the image file, if any
            if self.is_writing_image:
                cv2.imwrite(self._image_filename, frame)
                self._image_filename = None
                logging.info(f"Screenshot saved to {self._image_filename}")

            # Write to the video file, if any
            self._write_video_frame(frame)

            # Save the frame for external access
            with self._frame_lock:
                self._frame = frame

    @property
    def frame(self):
        """Get the current frame, mirrored if configured to do so."""
        with self._frame_lock:
            if self._frame is None:
                return None
                
            if self.should_mirror_preview:
                return numpy.fliplr(self._frame).copy()
            else:
                return self._frame

    @property
    def is_writing_image(self):
        """Check if an image is currently being written."""
        return self._image_filename is not None

    @property
    def is_writing_video(self):
        """Check if a video is currently being recorded."""
        return self._video_filename is not None

    def write_image(self, filename):
        """
        Schedule writing the next frame to an image file.
        
        Args:
            filename: The path to save the image to
        """
        self._image_filename = filename
        logging.info(f"Taking screenshot: {filename}")

    def start_writing_video(self, filename, 
                           encoding=cv2.VideoWriter_fourcc(*'MJPG')):
        """
        Start recording frames to a video file.
        
        Args:
            filename: The path to save the video to
            encoding: The FourCC code for the video codec
        """
        self._video_filename = filename
        self._video_encoding = encoding
        logging.info(f"Started recording video: {filename}")

    def stop_writing_video(self):
        """Stop recording video."""
        if self.is_writing_video:
            logging.info(f"Stopped recording video: {self._video_filename}")
            self._video_filename = None
            self._video_encoding = None
            self._video_writer = None

    def stop(self):
        """Stop the recorder thread and clean up resources."""
        self.stopped = True
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _write_video_frame(self, frame):
        """
        Write a frame to the video file if recording is active.
        
        Args:
            frame: The frame to write
        """
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS) if hasattr(self._capture, 'get') else 0
            
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate
                if self._frames_elapsed < 20:
                    # Wait until more frames elapse so that the estimate is more stable
                    return
                else:
                    fps = self._fps_estimate
            
            # Get frame size from the capture if possible, otherwise from the frame
            if hasattr(self._capture, 'get'):
                width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                height, width = frame.shape[:2]
                
            size = (width, height)
            self._video_writer = cv2.VideoWriter(
                self._video_filename, self._video_encoding,
                fps, size)

        self._video_writer.write(frame)