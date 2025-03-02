#\!/usr/bin/env python3

"""
Launcher script for Jarvis application.
"""

import sys
import os

# Add parent directory to the path for development mode
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jarvis import main

if __name__ == "__main__":
    main()
