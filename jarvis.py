#!/usr/bin/env python3


import logging
from threading import Thread

import cv2

import filters
import rects
from video_ui import PyQtWindowManager
from face_detector import FaceDetector
from streams import DummyStream, WebcamVideoStream, ThreadedWebStream
from video_recorder import VideoRecorder
from face_detection import Face


class Jarvis(object):
    def __init__(self):
        self._should_draw_debug = False
        self.raw_camera_stream = WebcamVideoStream(should_mirror=True)
        self.raw_web_stream = ThreadedWebStream(self.raw_camera_stream, port=8000)
        self.processed_camera_stream = DummyStream()
        self.processed_web_stream = ThreadedWebStream(self.processed_camera_stream, port=8888)
        self.face_detector = FaceDetector()
        self.window_manager = PyQtWindowManager('Jarvis - Computer Vision', self.on_key_press)
        self.window_manager.filterChanged.connect(self.on_filter_changed)
        self.window_manager.showFilteredChanged.connect(self.on_show_filtered_changed)
        self.video_recorder = VideoRecorder(self.raw_camera_stream)
        
        # Initialize filters
        self.current_filter = None
        self.filter_intensity = 50
        self.show_filtered_view = False
        self._initialize_filters()

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
        logging.info('Starting video recorder')
        self.video_recorder.start()

    def run(self):
        """Run the main loop."""
        try:
            self.start()
            self.window_manager.create_window()
            while self.window_manager.is_window_created:
                frame = self.raw_camera_stream.read()
                #frame = self.capture_manager.frame
                if frame is not None:
                    # Initialize frame skip counters if needed
                    if not hasattr(self, '_detection_interval'):
                        self._detection_interval = 0  # Counter for detection frequency
                        self._detection_frame_skip = 0  # Number of frames to skip
                    
                    # Adaptive frame skipping for face detection
                    # Once we have faces, we can skip more frames to smooth performance
                    if hasattr(self, '_smoothed_faces') and self._smoothed_faces is not None:
                        max_skip = 2  # Skip at most 2 frames when we have faces
                    else:
                        max_skip = 0  # Don't skip frames when searching for faces
                    
                    # Update face detection on some frames, not every frame
                    if self._detection_interval >= self._detection_frame_skip:
                        self.face_detector.update(frame)
                        current_faces = self.face_detector.faces
                        
                        # Reset counter and set next skip amount
                        self._detection_interval = 0
                        
                        # Adaptively set frame skip based on detection results
                        if current_faces and len(current_faces) > 0:
                            # We found faces, can skip more frames
                            self._detection_frame_skip = max_skip
                        else:
                            # No faces found, reduce skipping to find them faster
                            self._detection_frame_skip = 0
                    else:
                        # Skip face detection this frame, use previous results
                        self._detection_interval += 1
                        
                        # Use last known detection results
                        if hasattr(self, '_last_faces'):
                            current_faces = self._last_faces
                        else:
                            current_faces = []
                    
                    # Store current faces for skipped frames
                    self._last_faces = current_faces
                    
                    # Initialize history for temporal smoothing if needed
                    if not hasattr(self, '_face_history'):
                        self._face_history = []
                        self._smoothed_faces = None
                        self._frame_count = 0
                        self._previous_face_count = 0
                        self._stable_count = 0  # Count frames with stable detection
                    
                    # Initialize tracking variables
                    if not hasattr(self, '_frames_since_detection'):
                        self._frames_since_detection = 0
                        self._expected_face_count = 1  # Usually expect 1 face
                        self._face_count_confidence = 0
                        
                    # Get the current number of faces
                    current_face_count = len(current_faces) if current_faces is not None else 0
                        
                    # Track frames with no detection or unexpected count
                    if current_face_count == 0:
                        self._frames_since_detection += 1
                    elif current_face_count != self._expected_face_count:
                        # When we detect an unexpected number (like 2), 
                        # don't immediately trust it
                        if self._face_count_confidence < 5:
                            self._face_count_confidence += 1
                        else:
                            # After seeing the same count for 5+ frames, accept it
                            self._expected_face_count = current_face_count
                            self._face_count_confidence = 0
                            self._frames_since_detection = 0
                    else:
                        # We see the expected number of faces
                        self._frames_since_detection = 0
                        self._face_count_confidence = 0
                    
                    # Filter the detected faces based on our confidence
                    filtered_faces = None
                    if current_face_count > 0:
                        if current_face_count == self._expected_face_count or self._face_count_confidence >= 5:
                            # Accept the faces if they match expectations or we've seen this count consistently
                            filtered_faces = current_faces
                        elif current_face_count > self._expected_face_count and self._expected_face_count == 1:
                            # If we're detecting extra faces (but expect 1),
                            # just take the largest face as it's likely the correct one
                            if current_faces and len(current_faces) > 0:
                                # Find the face with largest area
                                largest_face = max(current_faces, key=lambda face: 
                                                 (face.face_rect[2] * face.face_rect[3]) 
                                                 if face.face_rect is not None else 0)
                                filtered_faces = [largest_face]
                    
                    # Keep most recent reliable detection if we're only missing for a few frames
                    if self._frames_since_detection > 0 and self._frames_since_detection <= 15:
                        # Don't update face history for brief disappearances
                        pass
                    else:
                        # Update history with filtered faces
                        self._face_history.append(filtered_faces)
                        if len(self._face_history) > 10:  # Use a reasonable history window
                            self._face_history.pop(0)
                    
                    # Apply temporal smoothing to reduce jitter
                    # If we have filtered faces in this frame, apply smoothing
                    if filtered_faces is not None:
                        # If this is our first face detection, just use it directly
                        if self._smoothed_faces is None:
                            self._smoothed_faces = filtered_faces
                            # Initialize a history of face rectangles for each face
                            self._rect_history = [[] for _ in range(len(filtered_faces))]
                        else:
                            # We already have smoothed faces, update them
                            for i, face in enumerate(filtered_faces):
                                if i < len(self._smoothed_faces):
                                    # Apply smoothing only to the face rectangle
                                    if face.face_rect is not None and self._smoothed_faces[i].face_rect is not None:
                                        x, y, w, h = face.face_rect
                                        
                                        # Update rectangle history for this face
                                        if i >= len(self._rect_history):
                                            self._rect_history.append([])
                                        self._rect_history[i].append(face.face_rect)
                                        if len(self._rect_history[i]) > 8:  # Keep last 8 frames
                                            self._rect_history[i].pop(0)
                                        
                                        # Calculate smooth rectangle by averaging recent positions
                                        if len(self._rect_history[i]) >= 3:
                                            # Get average of recent rectangles
                                            avg_x = sum(rect[0] for rect in self._rect_history[i]) / len(self._rect_history[i])
                                            avg_y = sum(rect[1] for rect in self._rect_history[i]) / len(self._rect_history[i])
                                            avg_w = sum(rect[2] for rect in self._rect_history[i]) / len(self._rect_history[i])
                                            avg_h = sum(rect[3] for rect in self._rect_history[i]) / len(self._rect_history[i])
                                            
                                            # Create smoothed rectangle
                                            smoothed_rect = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
                                            
                                            # Apply smooth rectangle
                                            self._smoothed_faces[i].face_rect = smoothed_rect
                                        else:
                                            # Not enough history yet, just use current detection
                                            self._smoothed_faces[i].face_rect = face.face_rect
                                        
                                        # Copy other facial features (eyes, nose, mouth)
                                        self._smoothed_faces[i].left_eye_rect = face.left_eye_rect
                                        self._smoothed_faces[i].right_eye_rect = face.right_eye_rect
                                        self._smoothed_faces[i].nose_rect = face.nose_rect
                                        self._smoothed_faces[i].mouth_rect = face.mouth_rect
                                else:
                                    # We have a new face, add it
                                    self._smoothed_faces.append(face)
                                    if i >= len(self._rect_history):
                                        self._rect_history.append([])
                                    self._rect_history[i].append(face.face_rect)
                    
                    # Otherwise if we've only temporarily lost detection, keep using previous detection
                    elif self._frames_since_detection <= 15 and self._smoothed_faces is not None:
                        # Keep using the last detected faces for a brief period
                        pass  # _smoothed_faces stays the same
                    
                    # We've lost detection for too long, clear tracking
                    else:
                        self._smoothed_faces = None
                        self._rect_history = []
                    
                    # Get the stable face count
                    stable_face_count = len(self._smoothed_faces) if self._smoothed_faces is not None else 0
                    
                    # Increment frame counter
                    self._frame_count += 1
                    
                    # Log only when face count changes (after smoothing)
                    if stable_face_count != self._previous_face_count:
                        # Only log changes that persist for at least 3 frames
                        self._stable_count += 1
                        if self._stable_count >= 3:
                            if stable_face_count > self._previous_face_count:
                                logging.info(f"Frame {self._frame_count}: Detected {stable_face_count} face(s)")
                            else:
                                logging.info(f"Frame {self._frame_count}: Lost face detection - now {stable_face_count} face(s)")
                            self._previous_face_count = stable_face_count
                            self._stable_count = 0
                    else:
                        self._stable_count = 0
                    
                    # Pass the smoothed face data to the VideoDisplay widget
                    self.window_manager.video_display.set_faces(self._smoothed_faces)
                    
                    # Update the debug state in the UI
                    self.window_manager.video_display.set_debug_mode(self._should_draw_debug)
                    
                    # Create a copy of the frame for processing
                    processed_frame = frame.copy()
                    
                    # Apply filter to the processed frame if needed
                    if self.current_filter and self.current_filter != 'none':
                        # Apply the selected filter
                        self.apply_filter(processed_frame, processed_frame)
                    
                    # Apply face detection annotations to the processed frame
                    self._draw_face_annotations(processed_frame)
                    
                    # Update the processed stream with the processed frame
                    self.processed_camera_stream.frame = processed_frame
                    
                    # Decide which frame to show in the UI
                    if self.show_filtered_view and self.current_filter and self.current_filter != 'none':
                        # Show the filtered frame in the UI
                        display_frame = self.processed_camera_stream.frame
                    else:
                        # Show the raw frame in the UI
                        display_frame = frame
                        
                    # Send the selected frame to the UI for display
                    self.window_manager.show_frame(display_frame)
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
        # Stop the video recorder
        logging.info('Stopping video recorder')
        self.video_recorder.stop()
        # Stop the raw camera stream feed
        logging.info('Stopping raw camera stream')
        self.raw_camera_stream.stop()
        # Stop the dummy feed of processed frames
        logging.info('Stopping processed camera stream')
        self.processed_camera_stream.stop()

    def screenshot(self):
        self.video_recorder.capture_screenshot('screenshot.png')

    def toggle_record_video(self):
        if not self.video_recorder.is_writing_video:
            self.video_recorder.start_recording('screencast.avi')
        else:
            self.video_recorder.stop_recording()

    def toggle_show_detection(self):
        """Toggle debug display and update UI state."""
        self._should_draw_debug = not self._should_draw_debug
        # Sync the UI state with our internal state
        self.window_manager.video_display.set_debug_mode(self._should_draw_debug)
        
        # Block signals to prevent recursive callbacks
        self.window_manager.detection_action.blockSignals(True)
        self.window_manager.show_detection_cb.blockSignals(True)
        
        # Ensure the toolbar button and checkbox match our state
        self.window_manager.detection_action.setChecked(self._should_draw_debug)
        self.window_manager.show_detection_cb.setChecked(self._should_draw_debug)
        
        # Re-enable signals
        self.window_manager.detection_action.blockSignals(False)
        self.window_manager.show_detection_cb.blockSignals(False)

    def _initialize_filters(self):
        """Initialize the image filters."""
        # Create filter instances
        self.filters = {
            'none': None,
            'edges': filters.FindEdgesFilter(),
            'sharpen': filters.SharpenFilter(),
            'blur': filters.BlurFilter(),
            'emboss': filters.EmbossFilter(),
            'cross_process': filters.BGRCrossProcessCurveFilter(),
            'portra': filters.BGRPortraCurveFilter(),
            'provia': filters.BGRProviaCurveFilter(),
            'velvia': filters.BGRVelviaCurveFilter()
        }
    
    def apply_filter(self, src, dst):
        """Apply the currently selected filter to the frame."""
        if not self.current_filter or self.current_filter == 'none' or self.current_filter not in self.filters:
            # No filtering needed, just copy
            if src is not dst:
                dst[:] = src
            return
        
        filter_obj = self.filters[self.current_filter]
        
        # Special case for edges - use stroke_edges function instead of FindEdgesFilter
        if self.current_filter == 'edges':
            # For edge detection, we can adjust parameters based on intensity
            blur_size = max(3, 3 + (self.filter_intensity // 10) * 2)  # Odd values 3-13
            edge_size = max(3, 3 + (self.filter_intensity // 20) * 2)  # Odd values 3-7
            
            # Ensure odd values
            if blur_size % 2 == 0:
                blur_size += 1
            if edge_size % 2 == 0:
                edge_size += 1
                
            filters.stroke_edges(src, dst, blur_size, edge_size)
        else:
            # Apply the filter - convolution or curve filter
            filter_obj.apply(src, dst)
            
    def on_filter_changed(self, filter_id, intensity):
        """Handle filter change from UI."""
        logging.info(f"Filter changed to {filter_id} with intensity {intensity}")
        self.current_filter = filter_id
        self.filter_intensity = intensity
        
    def on_show_filtered_changed(self, show_filtered):
        """Handle change in whether to show filtered stream."""
        logging.info(f"Show filtered view changed to {show_filtered}")
        self.show_filtered_view = show_filtered
        
    def _draw_face_annotations(self, frame):
        """Draw face detection annotations on the given frame."""
        if not self._should_draw_debug or not hasattr(self, '_smoothed_faces') or not self._smoothed_faces:
            return
            
        # Import colours here to avoid circular imports
        import colours
        
        # Draw debug overlay
        for face in self._smoothed_faces:
            # Draw face rectangle with a more refined thickness
            if face.face_rect is not None:
                x, y, w, h = face.face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colours.FACE_COLOUR, 3)
            
            # Draw eye rectangles with more subtle thickness
            if face.left_eye_rect is not None:
                x, y, w, h = face.left_eye_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colours.LEFT_EYE_COLOUR, 2)
                
            if face.right_eye_rect is not None:
                x, y, w, h = face.right_eye_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colours.RIGHT_EYE_COLOUR, 2)
            
            # Draw nose and mouth rectangles with refined thickness
            if face.nose_rect is not None:
                x, y, w, h = face.nose_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colours.NOSE_COLOUR, 2)
                
            if face.mouth_rect is not None:
                x, y, w, h = face.mouth_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colours.MOUTH_COLOUR, 2)
        
        # Draw text for number of faces
        h, w = frame.shape[:2]
        found = f"Faces: {len(self._smoothed_faces)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_pos = int(w * 0.05)
        y_pos = int(h * 0.1)
        
        # Create a background box for better text readability
        text_size = cv2.getTextSize(found, font, 1.2, 2)[0]
        box_coords = ((x_pos-10, y_pos+10), (x_pos + text_size[0]+10, y_pos - text_size[1]-10))
        cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)  # Filled black background
        
        # Draw text with improved readability
        cv2.putText(frame, found, (x_pos, y_pos), font, 1.2, colours.TEXT_OUTLINE_COLOUR, 4, cv2.LINE_AA)
        cv2.putText(frame, found, (x_pos, y_pos), font, 1.2, colours.HIGHLIGHT_TEXT_COLOUR, 2, cv2.LINE_AA)
    
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
    logging.basicConfig(level=logging.DEBUG)
    Jarvis().run()