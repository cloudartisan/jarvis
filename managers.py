#!/usr/bin/env python3

"""
DEPRECATED: This module is kept for backward compatibility only.

The functionality has been moved to more appropriately named modules:
- ThreadedCaptureManager -> MediaRecorder in streams.py
- Window managers -> PyQtWindowManager in video_ui.py 
- Web streaming -> ThreadedWebStream in streams.py

Please update your imports to use the new modules directly.
"""

import warnings
import logging
from streams import MediaRecorder

# Show deprecation warning
warnings.warn(
    "The managers module is deprecated. Please use streams.MediaRecorder, "
    "video_ui.PyQtWindowManager, and streams.ThreadedWebStream instead.",
    DeprecationWarning, stacklevel=2
)

# For backward compatibility
class ThreadedCaptureManager(MediaRecorder):
    """
    DEPRECATED: Use MediaRecorder from streams.py instead.
    
    This class is kept for backward compatibility only.
    """
    def __init__(self, capture, should_mirror_capture=False, 
                 preview_manager=None, should_mirror_preview=False):
        """Initialize with backward compatible parameters."""
        logging.warning("ThreadedCaptureManager is deprecated. Use MediaRecorder instead.")
        super(ThreadedCaptureManager, self).__init__(
            capture=capture,
            should_mirror_preview=should_mirror_preview
        )