import cv2
import time
import pandas as pd


vidcap = cv2.VideoCapture('ACT_13_1_4.mp4')
# frame_width = int(vidcap.get(3))
# frame_height = int(vidcap.get(4))
frame_width = 1920
frame_height = 1080
size = (frame_width, frame_height)

result = cv2.VideoWriter('violence.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)

success, frame = vidcap.read()
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
df = pd.read_csv('inferenceACT1314.csv')
frame_no = 0
id_no = 0


while True:
    print(success)
    if success == True:

        # if success:
        infer = df.loc[df['image_id'] == frame_no]
        #print("type of infer: ", type(infer))
        frame_no = frame_no + 1
        color = (0, 0, 255)
        #print("look for me")
        #print(infer)
        #time.sleep(6)
        for i in range(len(infer)):
            #print("this is id ::::::", id_no)
            x1 = int(infer.loc[id_no, "box0"])
            y1 = int(infer.loc[id_no, "box1"])
            x2 = x1 + int(infer.loc[id_no, "box2"])
            y2 = y1 + int(infer.loc[id_no, "box3"])
            #print(infer.loc[id_no, "image_id"], infer.loc[id_no, "idx"], infer.loc[id_no, "action"])
            if(int(infer.loc[id_no, "action"]) == 1):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)
                cv2.putText(frame, "Violent", (x1, y1), font, 1, color, 3, cv2.LINE_AA)

            id_no = id_no + 1
            #time.sleep(6)
        cv2.imshow("output", frame)
        result.write(frame)
        success, frame = vidcap.read()
        if cv2.waitKey(1) & 0xFF == ord('s'):
          break
            #cv2.waitKey(0)
    else:
        vidcap.release()
        result.release()
        cv2.destroyAllWindows()
        break
vidcap.release()
result.release()
cv2.destroyAllWindows()

# while success:
#   #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#   success, frame = vidcap.read()
#   cv2.rectangle()
#   print('Read a new frame: ', success)
#   count += 1
