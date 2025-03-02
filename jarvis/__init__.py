"""
Jarvis: Computer Vision Application

A modular Python application for face detection, video processing,
and image filtering.
"""

__version__ = '0.1.0'

from jarvis.core.app import Jarvis

def main():
    """Run the Jarvis application."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app = Jarvis()
    app.run()
