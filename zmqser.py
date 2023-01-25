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
#cap = CamGear(source='udpsrc port=5000 caps = "application/x-rtp, media=video,payload=96,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
cap = CamGear(source='udpsrc port=5000 caps = "application/x-rtp, media=video,payload=96,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
###cap = CamGear(source='udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5200 ! application/x-rtp, media=video, clock-rate=90000, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink').start()
########cap = CamGear(source = 'udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink').start()
# regular feed
####cap = CamGear(source='udpsrc port=5000 caps = "application/x-rtp, media=video, clock-rate=(int)90000, payload=96,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
producer = FrameProducer("5565")
exp = producer.connect()
print("zmq server")
print(exp)
#cap = cv2.VideoCapture("video.mp4")
#cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()
    #time.sleep(0.3)

    producer.send(frame)
    print("frame sent")
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# cap.release()
# cv2.destroyAllWindows()
