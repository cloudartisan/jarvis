#!/usr/bin/env python


import time
import StringIO
from threading import Thread
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn

import cv2
import numpy
from PIL import Image


class DummyStream:
    def __init__(self):
        self.stopped = False
        self._frame = None

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    def read(self):
        return self.frame

    def start(self):
        return self

    def stop(self):
        self.stopped = True


class WebcamVideoStream:
    def __init__(self, device=0, should_mirror=False):
        self.should_mirror = should_mirror
        self.stopped = False
        self.grabbed = False
        self._frame = None
        self._stream = cv2.VideoCapture(device)

    def read(self):
        if self.should_mirror and self._frame is not None:
            return numpy.fliplr(self._frame).copy()
        else:
            return self._frame

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self._frame = self._stream.read()

    def stop(self):
        self.stopped = True


class WebRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while not self.server.stopped:
                frame = self.server.camera_feed.read()
                if frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                s = StringIO.StringIO()
                image.save(s, 'JPEG')
                self.wfile.write('--jpgboundary')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(s.len))
                self.end_headers()
                image.save(self.wfile, 'JPEG')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="/stream.mjpg"/>')
            self.wfile.write('</body></html>')


class ThreadedWebStream(Thread):
    def __init__(self, camera_feed, ip='127.0.0.1', port=8000):
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass
        super(ThreadedWebStream, self).__init__()
        self.ip = ip
        self.port = port
        self.server = ThreadedHTTPServer((ip, port), WebRequestHandler)
        self.server.camera_feed = camera_feed
        self.server.stopped = True

    def run(self):
        self.server.stopped = False
        self.server.serve_forever()

    def stop(self):
        self.server.stopped = True
        self.server.shutdown()

    def __str__(self):
        return "{}:{}".format(self.ip, self.port)
