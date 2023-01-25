import zmq
import random
import sys
import time
import cv2
import base64
import numpy as np
from FrameProducer import FrameProducer
import pafy

# from vidgear.gears import CamGear   https://www.youtube.com/watch?v=ud3IXvT1vhc https://www.youtube.com/watch?v=qhQD09Jy1xE
# cap = CamGear(source='udpsrc port=5000 caps = "application/x-rtp, media=video,payload=96,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
# regular feed
url = "https://www.youtube.com/watch?v=ud3IXvT1vhc"
video = pafy.new(url)
best = video.streams[1]
print("best is....",best)
producer = FrameProducer("5565")
exp = producer.connect()
print("yt server")
#print(exp)
cap = cv2.VideoCapture(best.url)

while True:
    ret, frame = cap.read()
    #time.sleep(0.3)
    if ret==False:
        print("no frame")
        cap = cv2.VideoCapture(best.url)
        continue
    producer.send(frame)
    ##print("frame sent")
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()
