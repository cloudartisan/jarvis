#!/usr/bin/env python3


import logging
from threading import Thread

import cv2

import filters
import rects
from managers import TkinterWindowManager, ThreadedCaptureManager
from trackers import FaceTracker
from streams import DummyStream, WebcamVideoStream, ThreadedWebStream


class Jarvis(object):
    def __init__(self):
        self._should_draw_debug = False
        self.raw_camera_stream = WebcamVideoStream(should_mirror=True)
        self.raw_web_stream = ThreadedWebStream(self.raw_camera_stream, port=8000)
        self.processed_camera_stream = DummyStream()
        self.processed_web_stream = ThreadedWebStream(self.processed_camera_stream, port=8888)
        self.face_tracker = FaceTracker()
        self.window_manager = TkinterWindowManager('Jarvis', self.on_key_press)
        self.capture_manager = ThreadedCaptureManager(self.raw_camera_stream,
                False, None, False)

    def start(self):
        logging.info('Starting raw camera stream')
        self.raw_camera_stream.start()
        logging.info("Starting raw web stream {}".format(
            self.raw_web_stream))
        self.raw_web_stream.start()
        logging.info('Starting processed camera stream')
        self.processed_camera_stream.start()
        logging.info("Starting processed web stream {}".format(
            self.processed_web_stream))
        self.processed_web_stream.start()
        logging.info('Starting capture manager')
        self.capture_manager.start()

    def run(self):
        """Run the main loop."""
        try:
            self.start()
            self.window_manager.create_window()
            while self.window_manager.is_window_created:
                frame = self.raw_camera_stream.read()
                #frame = self.capture_manager.frame
                if frame is not None:
                    self.face_tracker.update(frame)
                    faces = self.face_tracker.faces
                    if faces is not None:
                        logging.debug("Found {} faces".format(len(faces)))
                    if self._should_draw_debug:
                        self.face_tracker.draw_debug_text(frame)
                        self.face_tracker.draw_debug_rects(frame)
                    self.processed_camera_stream.frame = frame
                    self.window_manager.show(frame)
                self.window_manager.process_events()
        finally:
            self.stop()

    def stop(self):
        # Stop web streaming from the raw camera feed
        logging.info('Stopping raw web stream')
        self.raw_web_stream.stop()
        self.raw_web_stream.join()
        # Stop web streaming the processed frames
        logging.info('Stopping processed web stream')
        self.processed_web_stream.stop()
        self.processed_web_stream.join()
        # Stop the capture manager
        logging.info('Stopping capture manager')
        self.capture_manager.stop()
        # Stop the raw camera stream feed
        logging.info('Stopping raw camera stream')
        self.raw_camera_stream.stop()
        # Stop the dummy feed of processed frames
        logging.info('Stopping processed camera stream')
        self.processed_camera_stream.stop()

    def screenshot(self):
        self.capture_manager.write_image('screenshot.png')

    def toggle_record_video(self):
        if not self.capture_manager.is_writing_video:
            self.capture_manager.start_writing_video('screencast.avi')
        else:
            self.capture_manager.stop_writing_video()

    def toggle_show_detection(self):
        self._should_draw_debug = not self._should_draw_debug

    def on_key_press(self, keycode):
        """Handle a key press.

        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        x      -> Start/stop drawing debug data.
        escape -> Quit.
        """
        if keycode == 32: # space
            self.screenshot()
        elif keycode == 9: # tab
            self.toggle_record_video()
        elif keycode == 120: # x
            self.toggle_show_detection()
        elif keycode == 27: # escape
            self.window_manager.destroy_window()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Jarvis().run()