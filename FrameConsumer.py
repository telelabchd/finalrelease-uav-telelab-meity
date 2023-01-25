import zmq
import random
import sys
import time
import cv2
import base64
import numpy as np


class FrameConsumer():
    def __init__(self, port: str):
        self.port = port
        self.socket = None

    def connect(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        consum = self.socket.connect('tcp://192.168.100.143:' + self.port)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        return consum

    def recv(self):
        #print("inside receive")
        frame = self.socket.recv()

        #print(type(frame))
        img = base64.b64decode(frame)
        npimg = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        return source
