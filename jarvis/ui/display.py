#!/usr/bin/env python3

import logging
import sys
import cv2
import numpy as np
from jarvis.utils import colours
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, 
    QVBoxLayout, QHBoxLayout, QAction, QToolBar,
    QStatusBar, QDockWidget, QSlider, QCheckBox,
    QComboBox, QGridLayout, QGroupBox
)


class VideoDisplay(QLabel):
    """Widget for displaying video frames with proper scaling."""
    
    def __init__(self, parent=None):
        super(VideoDisplay, self).__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._original_frame = None
        self._debug_mode = False
        self._faces = []
        
    def set_debug_mode(self, enabled):
        """Enable or disable debug overlay"""
        self._debug_mode = enabled
        
    def set_faces(self, faces):
        """Set detected faces for debug overlay"""
        self._faces = faces
        
    def display_frame(self, frame):
        """Display a frame (numpy array) with proper scaling."""
        if frame is None:
            return
            
        # Save the original frame as a working copy
        display_frame = frame.copy()
        
        # Draw debug overlay if enabled
        if self._debug_mode and self._faces:
            # Draw with thick lines for better visibility
            for face in self._faces:
                # Draw face rectangle with a more refined thickness
                if face.face_rect is not None:
                    x, y, w, h = face.face_rect
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours.FACE_COLOUR, 3)
                
                # Draw eye rectangles with more subtle thickness
                if face.left_eye_rect is not None:
                    x, y, w, h = face.left_eye_rect
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours.LEFT_EYE_COLOUR, 2)
                    
                if face.right_eye_rect is not None:
                    x, y, w, h = face.right_eye_rect
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours.RIGHT_EYE_COLOUR, 2)
                
                # Draw nose and mouth rectangles with refined thickness
                if face.nose_rect is not None:
                    x, y, w, h = face.nose_rect
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours.NOSE_COLOUR, 2)
                    
                if face.mouth_rect is not None:
                    x, y, w, h = face.mouth_rect
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours.MOUTH_COLOUR, 2)
            
            # Draw text for number of faces
            h, w = display_frame.shape[:2]
            found = "Faces: {}".format(len(self._faces))
            font = cv2.FONT_HERSHEY_SIMPLEX
            x_pos = int(w * 0.05)
            y_pos = int(h * 0.1)
            # Create a background box for better text readability
            text_size = cv2.getTextSize(found, font, 1.2, 2)[0]
            box_coords = ((x_pos-10, y_pos+10), (x_pos + text_size[0]+10, y_pos - text_size[1]-10))
            cv2.rectangle(display_frame, box_coords[0], box_coords[1], (0, 0, 0), -1)  # Filled black background
            
            # Draw text with improved readability
            cv2.putText(display_frame, found, (x_pos, y_pos), font, 1.2, colours.TEXT_OUTLINE_COLOUR, 4, cv2.LINE_AA)
            cv2.putText(display_frame, found, (x_pos, y_pos), font, 1.2, colours.HIGHLIGHT_TEXT_COLOUR, 2, cv2.LINE_AA)
            
        # Convert BGR to RGB format
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Convert to QImage
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and scale to fit the widget while preserving aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.width(), self.height(), 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class PyQtWindowManager(QMainWindow):
    """Main application window and video display for Jarvis."""
    
    keyPressed = pyqtSignal(int)
    filterChanged = pyqtSignal(str, int)  # Filter name, intensity
    showFilteredChanged = pyqtSignal(bool)  # Whether to show filtered stream
    
    def __init__(self, window_name, key_press_callback=None):
        """Initialize the window manager."""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        super(PyQtWindowManager, self).__init__()
        
        self.setWindowTitle(window_name)
        self.resize(800, 600)
        
        # Set up key handling
        self.key_press_callback = key_press_callback
        self.keyPressed.connect(self._handle_key)
        
        # Set up central widget for video display
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create video display
        self.video_display = VideoDisplay()
        self.layout.addWidget(self.video_display)
        
        # Create menu and toolbar
        self._create_menu()
        self._create_toolbar()
        self._create_statusbar()
        
        # Create control panel
        self._create_control_panel()
        
        # Initialize filter state
        self.current_filter = "none"
        self.current_intensity = 50  # Store intensity as a value, not a widget
        
        self._is_window_created = False
        
    def _create_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Screenshot action
        screenshot_action = QAction('&Take Screenshot', self)
        screenshot_action.setShortcut('Space')
        screenshot_action.setStatusTip('Take a screenshot')
        screenshot_action.triggered.connect(self._on_screenshot)
        file_menu.addAction(screenshot_action)
        
        # Video recording action
        record_action = QAction('&Record Video', self)
        record_action.setShortcut('Tab')
        record_action.setStatusTip('Start/stop recording video')
        record_action.triggered.connect(self._on_toggle_record)
        file_menu.addAction(record_action)
        
        # Exit action
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Esc')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Show detection action
        detection_action = QAction('&Show Detection', self)
        detection_action.setShortcut('X')
        detection_action.setStatusTip('Show/hide detection data')
        detection_action.setCheckable(True)
        detection_action.triggered.connect(self._on_toggle_detection)
        view_menu.addAction(detection_action)
        
        # Filtered view action
        filtered_view_action = QAction('Show &Filtered Stream', self)
        filtered_view_action.setStatusTip('Toggle between raw and filtered video')
        filtered_view_action.setCheckable(True)
        filtered_view_action.toggled.connect(self._on_toggle_filtered_view)
        view_menu.addAction(filtered_view_action)
        self.filtered_view_action = filtered_view_action
        
        # Filters submenu
        filters_menu = menubar.addMenu('&Filters')
        
        # No filter action
        none_filter_action = QAction('&None', self)
        none_filter_action.setData('none')
        none_filter_action.triggered.connect(lambda: self._select_filter_from_menu('none'))
        filters_menu.addAction(none_filter_action)
        
        # Edge detection action
        edges_filter_action = QAction('&Edge Detection', self)
        edges_filter_action.setData('edges')
        edges_filter_action.triggered.connect(lambda: self._select_filter_from_menu('edges'))
        filters_menu.addAction(edges_filter_action)
        
        # Sharpen action
        sharpen_filter_action = QAction('&Sharpen', self)
        sharpen_filter_action.setData('sharpen')
        sharpen_filter_action.triggered.connect(lambda: self._select_filter_from_menu('sharpen'))
        filters_menu.addAction(sharpen_filter_action)
        
        # Blur action
        blur_filter_action = QAction('&Blur', self)
        blur_filter_action.setData('blur')
        blur_filter_action.triggered.connect(lambda: self._select_filter_from_menu('blur'))
        filters_menu.addAction(blur_filter_action)
        
        # Emboss action
        emboss_filter_action = QAction('E&mboss', self)
        emboss_filter_action.setData('emboss')
        emboss_filter_action.triggered.connect(lambda: self._select_filter_from_menu('emboss'))
        filters_menu.addAction(emboss_filter_action)
        
        # Separator for colour filters
        filters_menu.addSeparator()
        
        # Colour filters submenu
        colour_filters_menu = filters_menu.addMenu('Colour Filters')
        
        # Cross Process action
        cross_filter_action = QAction('Cross Process', self)
        cross_filter_action.setData('cross_process')
        cross_filter_action.triggered.connect(lambda: self._select_filter_from_menu('cross_process'))
        colour_filters_menu.addAction(cross_filter_action)
        
        # Portra action
        portra_filter_action = QAction('Portra', self)
        portra_filter_action.setData('portra')
        portra_filter_action.triggered.connect(lambda: self._select_filter_from_menu('portra'))
        colour_filters_menu.addAction(portra_filter_action)
        
        # Provia action
        provia_filter_action = QAction('Provia', self)
        provia_filter_action.setData('provia')
        provia_filter_action.triggered.connect(lambda: self._select_filter_from_menu('provia'))
        colour_filters_menu.addAction(provia_filter_action)
        
        # Velvia action
        velvia_filter_action = QAction('Velvia', self)
        velvia_filter_action.setData('velvia')
        velvia_filter_action.triggered.connect(lambda: self._select_filter_from_menu('velvia'))
        colour_filters_menu.addAction(velvia_filter_action)
    
    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar('Main Toolbar')
        self.addToolBar(toolbar)
        
        # Screenshot action
        screenshot_action = QAction('Screenshot', self)
        screenshot_action.triggered.connect(self._on_screenshot)
        toolbar.addAction(screenshot_action)
        
        # Record action
        self.record_action = QAction('Record', self)
        self.record_action.setCheckable(True)
        self.record_action.triggered.connect(self._on_toggle_record)
        toolbar.addAction(self.record_action)
        
        # Detection action
        self.detection_action = QAction('Detection', self)
        self.detection_action.setCheckable(True)
        self.detection_action.triggered.connect(self._on_toggle_detection)
        toolbar.addAction(self.detection_action)
        # Set initial checked state to false to match Jarvis state
        self.detection_action.setChecked(False)
    
    def _create_statusbar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage('Ready')
    
    def _create_control_panel(self):
        """Create the control panel as a dock widget."""
        # Create dock widget
        control_dock = QDockWidget("Controls", self)
        control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create dock content
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        
        # Face detection group
        face_group = QGroupBox("Face Detection")
        face_layout = QGridLayout()
        
        # Show detection checkbox 
        self.show_detection_cb = QCheckBox("Show Detection")
        self.show_detection_cb.toggled.connect(self._on_toggle_detection_checkbox)
        face_layout.addWidget(self.show_detection_cb, 0, 0)
        
        face_group.setLayout(face_layout)
        dock_layout.addWidget(face_group)
        
        # Video controls group
        video_group = QGroupBox("Video Controls")
        video_layout = QGridLayout()
        
        # Screenshot button
        screenshot_btn = QPushButton("Take Screenshot")
        screenshot_btn.clicked.connect(self._on_screenshot)
        video_layout.addWidget(screenshot_btn, 0, 0)
        
        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self._on_toggle_record)
        video_layout.addWidget(self.record_btn, 1, 0)
        
        video_group.setLayout(video_layout)
        dock_layout.addWidget(video_group)
        
        # Image filters group
        filter_group = QGroupBox("Image Filters")
        filter_layout = QGridLayout()
        
        # Filter selection dropdown
        filter_layout.addWidget(QLabel("Filter:"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("None", "none")
        self.filter_combo.addItem("Edge Detection", "edges")
        self.filter_combo.addItem("Sharpen", "sharpen")
        self.filter_combo.addItem("Blur", "blur")
        self.filter_combo.addItem("Emboss", "emboss")
        self.filter_combo.addItem("Cross Process", "cross_process")
        self.filter_combo.addItem("Portra", "portra")
        self.filter_combo.addItem("Provia", "provia")
        self.filter_combo.addItem("Velvia", "velvia")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_combo, 0, 1)
        
        # Filter intensity slider (for some filters)
        filter_layout.addWidget(QLabel("Intensity:"), 1, 0)
        self.filter_intensity = QSlider(Qt.Horizontal)
        self.filter_intensity.setMinimum(0)
        self.filter_intensity.setMaximum(100)
        self.filter_intensity.setValue(50)
        self.filter_intensity.valueChanged.connect(self._on_intensity_changed)
        filter_layout.addWidget(self.filter_intensity, 1, 1)
        
        # Checkbox to show filtered stream in the main UI
        self.show_filtered_cb = QCheckBox("Show Filtered Stream")
        self.show_filtered_cb.setChecked(False)
        self.show_filtered_cb.toggled.connect(self._on_toggle_filtered_view)
        filter_layout.addWidget(self.show_filtered_cb, 2, 0, 1, 2)
        
        # Add note about web stream
        filter_note = QLabel("Note: Filtered stream available at port 8888")
        filter_note.setWordWrap(True)
        filter_layout.addWidget(filter_note, 3, 0, 1, 2)
        
        filter_group.setLayout(filter_layout)
        dock_layout.addWidget(filter_group)
        
        # Add stretch to push controls to the top
        dock_layout.addStretch()
        
        # Set the dock widget's content
        control_dock.setWidget(dock_content)
        
        # Add dock widget to main window
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)
    
    def _on_screenshot(self):
        """Handle screenshot action."""
        if self.key_press_callback:
            self.key_press_callback(32)  # Space key code
        self.statusbar.showMessage('Screenshot taken', 2000)
    
    def _on_toggle_record(self, checked=None):
        """Handle toggle record action."""
        if self.key_press_callback:
            self.key_press_callback(9)  # Tab key code
            
        # Update button text based on state
        recording = self.record_action.isChecked()
        if recording:
            self.record_btn.setText("Stop Recording")
            self.statusbar.showMessage('Recording started')
        else:
            self.record_btn.setText("Start Recording")
            self.statusbar.showMessage('Recording stopped')
    
    def _on_toggle_detection(self, checked=None):
        """Handle toggle detection action from toolbar/menu."""
        # Block signals to prevent infinite recursion
        self.detection_action.blockSignals(True)
        
        if self.key_press_callback:
            # Force trigger the keypress callback with 'x' key code
            self.key_press_callback(120)
        
        # Show UI feedback
        showing = self.detection_action.isChecked()
        if showing:
            self.statusbar.showMessage('Detection data shown')
        else:
            self.statusbar.showMessage('Detection data hidden')
            
        # Re-enable signals
        self.detection_action.blockSignals(False)
            
    def _on_toggle_detection_checkbox(self, checked=None):
        """Handle toggle detection action from checkbox."""
        # Prevent signals from causing infinite recursion
        self.show_detection_cb.blockSignals(True)
        
        if self.key_press_callback:
            # Force trigger the keypress callback with 'x' key code
            self.key_press_callback(120)
        
        # Re-enable signals
        self.show_detection_cb.blockSignals(False)
    
    def _on_filter_changed(self, index):
        """Handle filter selection change."""
        # Get the filter ID from the current index
        filter_id = self.filter_combo.currentData()
        self.current_filter = filter_id
        
        # Enable/disable intensity slider based on filter type if it exists
        if hasattr(self, 'filter_intensity') and isinstance(self.filter_intensity, QSlider):
            self.filter_intensity.setEnabled(filter_id != "none")
        
        # Emit signal that filter changed
        self.filterChanged.emit(filter_id, self.current_intensity)
        
        # Update status bar
        filter_name = self.filter_combo.currentText()
        self.statusbar.showMessage(f'Filter set to {filter_name}', 2000)
    
    def _on_intensity_changed(self, value):
        """Handle filter intensity slider change."""
        self.current_intensity = value
        self.filterChanged.emit(self.current_filter, value)
    
    def _on_toggle_filtered_view(self, checked):
        """Handle toggle filtered view checkbox."""
        # Block signals to prevent recursive callbacks
        if hasattr(self, 'show_filtered_cb'):
            self.show_filtered_cb.blockSignals(True)
        if hasattr(self, 'filtered_view_action'):
            self.filtered_view_action.blockSignals(True)
            
        # Update checkbox and menu item state
        if hasattr(self, 'show_filtered_cb'):
            self.show_filtered_cb.setChecked(checked)
        if hasattr(self, 'filtered_view_action'):
            self.filtered_view_action.setChecked(checked)
            
        # Emit signal that filtered view changed
        self.showFilteredChanged.emit(checked)
        
        # Update status bar
        if checked:
            self.statusbar.showMessage('Showing filtered stream')
        else:
            self.statusbar.showMessage('Showing raw stream')
            
        # Re-enable signals
        if hasattr(self, 'show_filtered_cb'):
            self.show_filtered_cb.blockSignals(False)
        if hasattr(self, 'filtered_view_action'):
            self.filtered_view_action.blockSignals(False)
    
    def _select_filter_from_menu(self, filter_id):
        """Handle filter selection from menu."""
        # Update combo box to match menu selection
        index = self.filter_combo.findData(filter_id)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)
    
    def _handle_key(self, keycode):
        """Handle key press signal."""
        if self.key_press_callback:
            # Only log function keys, not normal keyboard input
            if keycode in [9, 27, 32, 120]:  # tab, esc, space, x
                logging.debug(f'UI key pressed: {keycode}')
            self.key_press_callback(keycode)
    
    def keyPressEvent(self, event):
        """Handle key press event."""
        key = event.key()
        
        # Map Qt key codes to ASCII for compatibility with existing code
        # Space -> 32, Tab -> 9, X -> 120, Escape -> 27
        if key == Qt.Key_Space:
            self.keyPressed.emit(32)
        elif key == Qt.Key_Tab:
            self.keyPressed.emit(9)
        elif key == Qt.Key_X:
            self.keyPressed.emit(120)
        elif key == Qt.Key_Escape:
            self.keyPressed.emit(27)
        else:
            # Try to get ASCII value for other keys if possible
            text = event.text()
            if text:
                self.keyPressed.emit(ord(text[0]))
    
    def create_window(self):
        """Create and show the window."""
        super(PyQtWindowManager, self).show()
        self._is_window_created = True
    
    def show_frame(self, frame):
        """Show a frame in the window."""
        self.video_display.display_frame(frame)
    
    def destroy_window(self):
        """Close the window."""
        self.close()
        self._is_window_created = False
    
    def process_events(self):
        """Process pending events."""
        QApplication.processEvents()
    
    @property
    def is_window_created(self):
        """Check if window is created."""
        return self._is_window_created
        

# QApplication requires QWidget, adding a simple import
from PyQt5.QtWidgets import QPushButton