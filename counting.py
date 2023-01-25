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
consum = consumer.connect()
print("ml server consumer")
print(consum)
producer = FrameProducer("5567")
exp = producer.connect()
print("ml server producer")
print(exp)
model = YOLO("vis1280yolov5.pt")
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
        #print(df, "random")
        num_people = len(df[df['class'] == 0]) #+ len(df['class'] == 1)
        cv2.putText(frame, "Number of people"+str(num_people), (10, 20), font, fontScale,color, thickness, cv2.LINE_4)
        #print("great",num_people)
        num_vechile = len(df[df['class'] == 3]) + len(df[df['class'] == 4])
        cv2.putText(frame, "Number of car" + str(num_vechile), (10, 40), font, fontScale, (255,0,0), thickness,cv2.LINE_4)
        #print("vechile",num_vechile)
        for i in range(df.shape[0]):
            #print("iiiii", i)
            if df.at[i, 'class'] == 0:
                a=2
                # cv2.putText(frame, 'p', (int(df.at[i, 'xmin']), int(df.at[i, 'ymin']) + 20), font,
                #             fontScale, color, thickness, cv2.LINE_4)
                cv2.circle(frame, (int(df.at[i, 'xmin']), int(df.at[i, 'ymin']) + 20), radius=0, color=(0, 0, 255), thickness=5)
            else:
                b=3
                # cv2.putText(frame, 'v', (int(df.at[i, 'xmin']), int(df.at[i, 'ymin']) + 20), font,
                #             fontScale, color, thickness, cv2.LINE_4)
                cv2.circle(frame, (int(df.at[i, 'xmin']), int(df.at[i, 'ymin']) + 20), radius=0, color=(255, 0, 0), thickness=5)
            #cv2.putText(frame, df.at[i, 'name'], (int(df.at[i, 'xmin']), int(df.at[i, 'ymin'])+20), font, fontScale, color,thickness, cv2.LINE_4)
    cv2.imshow("image", frame)
    producer.send(frame)
    cv2.waitKey(1)
# print ("Average messagedata value for topic '%s' was %dF" % (topicfilter, total_value / update_nbr))
