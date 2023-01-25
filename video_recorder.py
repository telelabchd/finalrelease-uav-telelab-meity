import zmq
import random
import sys
import time
import cv2
import base64
import numpy as np
from FrameProducer import FrameProducer

from vidgear.gears import CamGear
print("stuck here")
cap = CamGear(source='udpsrc port=5000 caps = "application/x-rtp, media=video,payload=96,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
# regular feed

producer = FrameProducer("5565")
exp = producer.connect()
print("zmq server")
print(exp)
#cap = cv2.VideoCapture("video.mp4")
#cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 30, (640, 480))

while True:
    frame = cap.read()
    #time.sleep(0.3)

    producer.send(frame)
    print("frame sent")
    cv2.imshow('frame', frame)
    video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        video.release()
        cap.release()
        break
# cap.release()
# cv2.destroyAllWindows()
