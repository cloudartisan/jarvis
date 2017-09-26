#!/usr/bin/env python


from threading import Thread

import cv2

import filters
import rects
from managers import TkinterWindowManager, ThreadedWebStreamManager, WindowManager, CaptureManager
from trackers import FaceTracker


class JarvisVision(object):
    def __init__(self):
        self._window_manager = TkinterWindowManager(
            'Jarvis', self.on_key_press)
        self._capture_manager = CaptureManager(
            cv2.VideoCapture(0),
            should_mirror_capture=True,
            preview_manager=self._window_manager,
            should_mirror_preview=False)
        self._threaded_web_stream_manager = ThreadedWebStreamManager(
            self._capture_manager)
        self._face_tracker = FaceTracker()
        self._should_draw_debug = False
        self._curve_filter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        try:
            self._window_manager.create_window()
            self._threaded_web_stream_manager.start()
            while self._window_manager.is_window_created:
                self._capture_manager.enter_frame()
                frame = self._capture_manager.frame

                if frame is not None:
                    self._face_tracker.update(frame)
                    faces = self._face_tracker.faces

                    #rects.swap_rects(frame, frame,
                                    #[face.face_rect for face in faces])
                    #filters.stroke_edges(frame, frame)
                    #self._curve_filter.apply(frame, frame)

                    if self._should_draw_debug:
                        self._face_tracker.draw_debug_text(frame)
                        self._face_tracker.draw_debug_rects(frame)

                    self._capture_manager.processed_frame = frame

                self._capture_manager.exit_frame()
                self._window_manager.process_events()
        finally:
            self._threaded_web_stream_manager.stop()
            self._threaded_web_stream_manager.join()

    def screenshot(self):
        self._capture_manager.write_image('screenshot.png')

    def toggle_record_video(self):
        if not self._capture_manager.is_writing_video:
            self._capture_manager.start_writing_video('screencast.avi')
        else:
            self._capture_manager.stop_writing_video()

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
            self._window_manager.destroy_window()


if __name__ == '__main__':
    JarvisVision().run()
