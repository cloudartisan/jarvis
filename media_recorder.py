#!/usr/bin/env python3


import logging
import time
from threading import Thread, Lock

import cv2
import numpy


class VideoRecorder(Thread):
    """
    Records frames from a video stream to image files or video files.
    Works with any source that has a read() method returning video frames.
    """
    def __init__(self, source, should_mirror=False):
        """
        Initialize the video recorder.
        
        Args:
            source: A video source with a read() method that returns frames
            should_mirror: Whether to mirror the frames horizontally
        """
        super(VideoRecorder, self).__init__()
        self.daemon = True
        self.should_mirror = should_mirror
        self.stopped = False
        
        self._source = source
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
            frame = self._source.read()
            if frame is None:
                continue

            # Update the FPS estimate
            if self._frames_elapsed == 0:
                self._start_time = time.time()
            else:
                time_elapsed = time.time() - self._start_time
                self._fps_estimate = self._frames_elapsed / time_elapsed
            self._frames_elapsed += 1

            # Write to the image file, if requested
            if self.is_writing_image:
                image_path = self._image_filename
                cv2.imwrite(image_path, frame)
                self._image_filename = None
                logging.info(f"Screenshot saved to {image_path}")

            # Write to the video file, if recording
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
                
            if self.should_mirror:
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

    def capture_screenshot(self, filename):
        """
        Schedule writing the next frame to an image file.
        
        Args:
            filename: The path to save the image to
        """
        self._image_filename = filename
        logging.info(f"Taking screenshot: {filename}")

    def start_recording(self, filename, 
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

    def stop_recording(self):
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
            fps = self._source.get(cv2.CAP_PROP_FPS) if hasattr(self._source, 'get') else 0
            
            if fps <= 0.0:
                # The source's FPS is unknown so use an estimate
                if self._frames_elapsed < 20:
                    # Wait until more frames elapse so that the estimate is more stable
                    return
                else:
                    fps = self._fps_estimate
            
            # Get frame size from the source if possible, otherwise from the frame
            if hasattr(self._source, 'get'):
                width = int(self._source.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._source.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                height, width = frame.shape[:2]
                
            size = (width, height)
            self._video_writer = cv2.VideoWriter(
                self._video_filename, self._video_encoding,
                fps, size)

        self._video_writer.write(frame)


# For backward compatibility
class MediaRecorder(VideoRecorder):
    """Alias for VideoRecorder for backward compatibility."""
    
    def write_image(self, filename):
        """Legacy method, use capture_screenshot instead."""
        return self.capture_screenshot(filename)
        
    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc(*'MJPG')):
        """Legacy method, use start_recording instead."""
        return self.start_recording(filename, encoding)
        
    def stop_writing_video(self):
        """Legacy method, use stop_recording instead."""
        return self.stop_recording()