import sys
import cv2
import zmq
import base64
import numpy as np
from modelreturn import YOLO
import pandas
from FrameConsumer import FrameConsumer
from FrameProducer import FrameProducer

consumer = FrameConsumer("5566")
consum=consumer.connect()
print("ml server consumer")
print(consum)
producer = FrameProducer("5567")
exp=producer.connect()
print("ml server producer")
print(exp)
model = YOLO("yolov5m6.pt") #
while True:
    ##print("rech")
    frame = consumer.recv()
    ##print("rech")
    yolo_detections = model(frame)
    df = yolo_detections.pandas().xyxy[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    if not df.empty:
        print(df.shape[0], "random")
        for i in range(df.shape[0]):
            ##print("iiiii", i)
            cv2.putText(frame, df.at[i, 'name'], (int(df.at[i, 'xmin']), int(df.at[i, 'ymin'])), font, fontScale, color,thickness, cv2.LINE_4)
    cv2.imshow("image", frame)
    producer.send(frame)
    cv2.waitKey(1)
# print ("Average messagedata value for topic '%s' was %dF" % (topicfilter, total_value / update_nbr))
