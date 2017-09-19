#!/usr/bin/env python

import time

import cv2
import numpy

from PIL import Image
from PIL import ImageTk

import Tkinter as tk


class CaptureManager(object):
    def __init__(
            self, capture, preview_window_manager=None,
             should_mirror_preview=False):
        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview

        self._capture = capture
        self._channel = 0
        self._entered_frame = False
        self._frame = None
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = long(0)
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def is_writing_image(self):
        return self._image_filename is not None

    @property
    def is_writing_video(self):
        return self._video_filename is not None

    def enter_frame(self):
        """Capture the next frame, if any."""
        # But first, check that any previous frame was exited.
        assert not self._entered_frame, \
            'previous enter_frame() had no matching exit_frame()'

        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        """Draw to the window. Write to files. Release the frame."""
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._entered_frame = False
            return

        # Update the FPS estimate and related variables.
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate =  self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # Draw to the window, if any.
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        # Write to the image file, if any.
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None

        # Write to the video file, if any.
        self._write_video_frame()

        # Release the frame.
        self._frame = None
        self._entered_frame = False

    def write_image(self, filename):
        """Write the next exited frame to an image file."""
        self._image_filename = filename

    def start_writing_video(
            self, filename,
            encoding=cv2.VideoWriter_fourcc(*'MJPG')):
        """Start writing exited frames to a video file."""
        self._video_filename = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        """Stop writing exited frames to a video file."""
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._frames_elapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(
                        cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(
                self._video_filename, self._video_encoding,
                fps, size)

        self._video_writer.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, key_press_callback=None):
        self.key_press_callback = key_press_callback
        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if self.key_press_callback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.key_press_callback(keycode)


class TkinterWindowManager(WindowManager):
    """
    When holding a key down, multiple key press and key release events are fired in
    succession. Debouncing is implemented in order to squash these repeated events
    and know when the "real" KeyRelease and KeyPress events happen.
    """
    def __init__(self, window_name, key_press_callback=None):
        self._has_prev_key_release = None
        super(TkinterWindowManager, self).__init__(window_name, key_press_callback)

    def create_window(self):
        self.root = tk.Tk()
        self.panel = None
        self.root.wm_title(self._window_name)
        self.root.wm_protocol('WM_DELETE_WINDOW', self.destroy_window)
        self._is_window_created = True

    def show(self, frame):
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # if the panel is not None, we need to initialize it
        if self.panel is None:
            self.panel = tk.Label(image=image)
            self.panel.image = image
            self.panel.bind('<KeyRelease>', self._on_key_release_repeat)
            self.panel.bind('<KeyPress>', self._on_key_press_repeat)
            self.panel.pack(side='left', padx=10, pady=10)
            self.panel.focus_set()
        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

        self.root.update()

    def _on_key_release(self, event):
        self._has_prev_key_release = None
        print '_on_key_release', repr(event.char), ord(event.char)
        if self.key_press_callback is not None:
            self.key_press_callback(ord(event.char))

    def _on_key_press(self, event):
        print '_on_key_press', repr(event.char)

    def _on_key_release_repeat(self, event):
        self._has_prev_key_release = self.root.after_idle(self._on_key_release, event)
        print '_on_key_release_repeat', repr(event.char)

    def _on_key_press_repeat(self, event):
        if self._has_prev_key_release:
            self.root.after_cancel(self._has_prev_key_release)
            self._has_prev_key_release = None
            print '_on_key_press_repeat', repr(event.char)
        else:
            self._on_key_press(event)

    def destroy_window(self):
        self.root.quit()
        self._is_window_created = False
