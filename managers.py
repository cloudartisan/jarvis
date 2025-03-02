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
from PIL import ImageTk

import tkinter as tk


class ThreadedCaptureManager(object):
    def __init__(
            self, capture, should_mirror_capture=False,
            preview_manager=None, should_mirror_preview=False):
        self.should_mirror_capture = should_mirror_capture
        self.preview_manager = preview_manager
        self.should_mirror_preview = should_mirror_preview

        self.processed_frame = None

        self.stopped = False

        self._capture = capture
        self._frame = None
        self._frame_lock = Lock()
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0  # Changed from long(0)
        self._fps_estimate = None

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        self.stopped = True

    def update(self):
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

            # Draw to the window, if any
            if self.preview_manager is not None:
                if self.should_mirror_preview:
                    mirrored_frame = numpy.fliplr(frame).copy()
                    self.preview_manager.show(mirrored_frame)
                else:
                    self.preview_manager.show(frame)

            # Write to the image file, if any
            if self.is_writing_image:
                cv2.imwrite(self._image_filename, frame)
                self._image_filename = None

            # Write to the video file, if any
            self._write_video_frame()

            # Save the frame for external access
            self._frame = frame

    @property
    def frame(self):
        # Set the captured frame for external access
        if self.should_mirror_capture:
            return numpy.fliplr(self._frame).copy()
        else:
            return self._frame

    @property
    def is_writing_image(self):
        return self._image_filename is not None

    @property
    def is_writing_video(self):
        return self._video_filename is not None

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

class CaptureManager(object):
    def __init__(
            self, capture, should_mirror_capture=False,
            preview_manager=None, should_mirror_preview=False):
        self.should_mirror_capture = should_mirror_capture
        self.preview_manager = preview_manager
        self.should_mirror_preview = should_mirror_preview

        self.processed_frame = None

        self._capture = capture
        self._frame_lock = Lock()
        self._frame = None
        self._entered_frame = False
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0  # Changed from long(0)
        self._fps_estimate = None

    @property
    def frame(self):
        with self._frame_lock:
            if self._entered_frame and self._frame is None:
                _, self._frame = self._capture.retrieve()
                if self.should_mirror_capture:
                    self._frame = numpy.fliplr(self._frame).copy()
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
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # Draw to the window, if any.
        if self.preview_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.preview_manager.show(mirrored_frame)
            else:
                self.preview_manager.show(self._frame)

        # Write to the image file, if any.
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None

        # Write to the video file, if any.
        self._write_video_frame()

        # Release the frame.
        with self._frame_lock:
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


class WebRequestManager(BaseHTTPRequestHandler):
    camera_feed = None

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    frame = self.camera_feed.frame
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
                except KeyboardInterrupt:
                    break
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="/stream.mjpg"/>')
            self.wfile.write(b'</body></html>')


class ThreadedWebStreamManager(Thread):
    def __init__(self, camera_feed, ip='127.0.0.1', port=8000):
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass
        super(ThreadedWebStreamManager, self).__init__()
        WebRequestManager.camera_feed = camera_feed
        self.server = ThreadedHTTPServer((ip, port), WebRequestManager)

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


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
        
        # Make frame responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create frame container that will resize
        self.container = tk.Frame(self.root)
        self.container.grid(row=0, column=0, sticky="nsew")
        
        self._original_width = 0
        self._original_height = 0
        
        # Bind window resize event
        self.root.bind("<Configure>", self._on_window_resize)
        
        self._is_window_created = True

    def _on_window_resize(self, event):
        # Only process window events, not widget events
        if event.widget == self.root:
            width = event.width
            height = event.height
            logging.debug(f"Window resized to {width}x{height}")
    
    def show(self, frame):
        # Save original frame dimensions on first run
        if self._original_width == 0:
            self._original_width = frame.shape[1]
            self._original_height = frame.shape[0]
            
        # Get current window dimensions
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Only resize if window has been properly initialized
        if window_width > 100 and window_height > 100:
            # Calculate new dimensions while preserving aspect ratio
            display_width = window_width - 20  # Adjust for padding
            display_height = window_height - 20  # Adjust for padding
            
            # Determine scaling factor based on both dimensions
            width_ratio = display_width / self._original_width
            height_ratio = display_height / self._original_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(self._original_width * scale_factor)
            new_height = int(self._original_height * scale_factor)
            
            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # if the panel is not None, we need to initialize it
        if self.panel is None:
            self.panel = tk.Label(self.container, image=image)
            self.panel.image = image
            self.panel.bind('<KeyRelease>', self._on_key_release_repeat)
            self.panel.bind('<KeyPress>', self._on_key_press_repeat)
            self.panel.pack(fill=tk.BOTH, expand=True)
            self.panel.focus_set()
        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

        self.root.update()

    def _on_key_release(self, event):
        self._has_prev_key_release = None
        logging.debug('_on_key_release {}'.format(repr(event.char)))
        if self.key_press_callback is not None and event.char != '':
            keycode = ord(event.char)
            logging.info('Callback triggered by keycode {}'.format(keycode))
            self.key_press_callback(keycode)

    def _on_key_press(self, event):
        logging.debug('_on_key_press {}'.format(repr(event.char)))

    def _on_key_release_repeat(self, event):
        self._has_prev_key_release = self.root.after_idle(self._on_key_release, event)
        logging.debug('_on_key_release_repeat {}'.format(repr(event.char)))

    def _on_key_press_repeat(self, event):
        if self._has_prev_key_release:
            self.root.after_cancel(self._has_prev_key_release)
            self._has_prev_key_release = None
            logging.debug('_on_key_press_repeat {}'.format(repr(event.char)))
        else:
            self._on_key_press(event)

    def destroy_window(self):
        self.root.quit()
        self._is_window_created = False