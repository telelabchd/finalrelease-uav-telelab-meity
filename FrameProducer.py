import zmq
import random
import sys
import time
import cv2
import base64
import numpy as np


class FrameProducer():
    def __init__(self, port: str):
        self.port = port
        self.socket = None

    def connect(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        val = self.socket.connect('tcp://192.168.100.143:'+self.port)
        return val
    def send(self, frame):
        #print(type(frame))
        #frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (1280, 720))
        encoded, buffer = cv2.imencode('.jpg', frame)
        message = base64.b64encode(buffer)
        #print(type(message))
        serv=self.socket.send(message)
        #print(" sending frame")
        #print(serv)
