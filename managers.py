#!/usr/bin/env python3

"""
DEPRECATED MODULE: This file exists for backward compatibility.

The functionality has been moved to more appropriately named modules:
- Video recording -> VideoRecorder in media_recorder.py
- UI/Window management -> PyQtWindowManager in video_ui.py 
- Web streaming -> ThreadedWebStream in streams.py

This module is empty and may be removed in future releases.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "The managers module is deprecated and empty. Please use video_recorder.VideoRecorder, "
    "video_ui.PyQtWindowManager, and streams.ThreadedWebStream instead.",
    DeprecationWarning, stacklevel=2
)