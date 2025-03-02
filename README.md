# Jarvis: Computer Vision Project

Jarvis is a computer vision application that can detect faces, track features, apply image filters, and provide video streaming capabilities. Built with OpenCV, PyQt5, and Python 3.

## Setup and Installation

### Prerequisites

For macOS:
```bash
brew install opencv
brew install portaudio
brew install python-tk@3.13  # Required for the GUI
```

For other platforms, install OpenCV, PortAudio, and Python Tkinter using your package manager.

### Python Environment Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify OpenCV is properly installed:
```bash
python -c "import cv2; print(cv2.__version__)"
```

## Running the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the main application
python jarvis.py
```

You can also run the individual utility scripts:

```bash
# Make sure you're in the virtual environment first
source venv/bin/activate

# Face detection from webcam
python scripts/detect_face_stream.py

# Web streaming of webcam
python scripts/web_serve_stream.py

# Speech recognition (requires Google Cloud credentials)
# Note: You need to set up a Google Cloud account and enable the Speech-to-Text API first
# Then download service account credentials and set this environment variable:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
python scripts/transcribe_mic_stream.py
```

### Running Tests

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Set your Google Cloud credentials (required for speech recognition tests)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json

# Run the test script
python test/test.py
```

Note: The test script uses Google's Speech-to-Text API and requires credentials. If you don't have Google Cloud credentials, you won't be able to run the tests. This won't affect the main functionality of the application.

**Important:** Always make sure your virtual environment is activated before running any scripts or the main application. Otherwise, the required dependencies won't be available.

## Web Streaming

The main application (jarvis.py) provides two web streams:
- Raw camera feed: http://localhost:8000/
- Processed video feed: http://localhost:8888/

When running the separate web streaming script (scripts/web_serve_stream.py):
- Camera feed: http://localhost:8080/cam.mjpg

You can view these streams in any web browser or embed them in other applications.

## Controls

### Keyboard Shortcuts
When the application window is in focus:
- **Space**: Take a screenshot
- **Tab**: Start/stop recording a screencast
- **X**: Toggle debug view (shows face detection rectangles)
- **Escape**: Quit the application

### UI Controls
The application features a control panel with:
- Filter selector dropdown
- Filter intensity slider
- Toggle for displaying the filtered stream
- Controls for screenshots and video recording

## Features

- Face detection and tracking with:
  - Multi-stage detection (DNN + cascade)
  - Feature tracking (eyes, nose, mouth)
  - Temporal smoothing to reduce jitter
  - Adaptive frame skipping for better performance

- Real-time image processing with filters:
  - **Edge Detection**: Highlights outlines and boundaries in the image
  - **Sharpen**: Enhances details and makes the image crisper
  - **Blur**: Smooths out noise and reduces detail
  - **Emboss**: Creates a 3D relief effect highlighting edges
  - **Film Emulation Filters**:
    - **Cross Process**: High contrast with altered colours, mimicking cross-processed film
    - **Portra**: Warm, natural skin tones inspired by Kodak Portra film
    - **Provia**: Balanced, natural colours inspired by Fuji Provia film
    - **Velvia**: Vibrant, saturated colours inspired by Fuji Velvia film

- Dual streaming over HTTP:
  - Raw video stream
  - Processed/filtered stream
  
- Media creation:
  - Video recording
  - Screenshot capture

## Troubleshooting

1. **Camera access issues**:
   - Ensure your webcam is connected and functioning
   - Check camera permissions for your terminal app/Python
   - If using macOS, you may need to grant permission in System Settings → Privacy & Security → Camera

2. **ImportError: No module named X**:
   - Make sure you have installed all requirements and activated your virtual environment
   - Try installing the specific missing package: `pip install X`

3. **Port already in use**:
   - If ports 8000/8888 are already taken, modify the port numbers in the code

4. **Tkinter issues**:
   - If you get `ModuleNotFoundError: No module named '_tkinter'`, install the Python Tkinter package for your system
   - On macOS: `brew install python-tk@3.13`
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`

5. **Error: externally-managed-environment**:
   - Make sure you're using a virtual environment: `python -m venv venv && source venv/bin/activate`

6. **Google Cloud Speech API errors**:
   - The transcribe_mic_stream.py script requires Google Cloud credentials
   - You need to create a Google Cloud account and enable the Speech-to-Text API
   - Create a service account key and download the JSON credentials file
   - Set the environment variable: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`
   - If you don't need speech recognition, you can ignore this step

## Project Structure

- `jarvis.py` - Main application file
- `managers.py` - Capture and streaming management
- `video_ui.py` - PyQt5 UI components and window management
- `face_detector.py` - Face and feature detection and tracking
- `streams.py` - Video stream abstractions
- `utils.py` - Helper functions
- `rects.py` - Rectangle handling functions
- `colours.py` - Colour constants for visual elements
- `filters.py` - Image processing filters and effects
- `face_detection/` - Package for different detection algorithms
  - `base.py` - Base classes for detection
  - `dnn_detector.py` - Deep neural network detector
  - `haar_detector.py` - Haar cascade detector
  - `face_recognition.py` - Face feature recognition
- `cascades/` - OpenCV face detection XML files
- `scripts/` - Utility scripts for face detection, web streaming, and speech transcription

## Using the Image Filters

1. Run the main application: `python jarvis.py`
2. Select a filter from the dropdown menu or Filters menu
3. Adjust the intensity using the slider
4. Toggle "Show Filtered Stream" to view the filtered video in the main window
5. View both streams simultaneously by opening these URLs in a browser:
   - Raw stream: http://localhost:8000/
   - Filtered stream: http://localhost:8888/