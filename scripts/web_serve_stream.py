#!/usr/bin/env python3

"""
A Simple mjpg stream http server
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from io import BytesIO
import time

import cv2
from PIL import Image


capture = None
image = None


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    rc, image = capture.read()
                    if not rc:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(image_rgb)
                    tmp_file = BytesIO()
                    jpg.save(tmp_file, 'JPEG')
                    self.wfile.write(b"--jpgboundary")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(tmp_file.getbuffer().nbytes))
                    self.end_headers()
                    try:
                        jpg.save(self.wfile, 'JPEG')
                        time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected
                        break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in stream loop: {e}")
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<image src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def main():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    capture.set(cv2.CAP_PROP_SATURATION,0.2)
    global image
    try:
        server = ThreadedHTTPServer(('localhost', 8080), CamHandler)
        print("server started")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()