import glob
import io
import pandas as pd

import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 0, 255)
thickness = 2

cap = cv2.VideoCapture('/media/telelab/volume1/Voilence Videos/ACT_13_1_2/ACT_13_1_2.mp4')  # '/media/telelab/30E3-03A0/Demo Videos/Person and Vehicle Counting/ACT_2_7.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('/media/telelab/volume1/Voilence Videos/ACT_13_1_2/act_13_1_2.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
files = [file for file in sorted(glob.glob("/media/telelab/volume1/Voilence Videos/ACT_13_1_2/gt_text/*"))]
for file_name in files:
    print(file_name)
    df = pd.read_csv(file_name, index_col=None, header=None)
    #print(df)
    #print("shape", df.shape[0])
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        print("frame is not read")
        break
    for i in range(df.shape[0]):

        # print("iiiii", i)
        #print(df.at[i, 0])
        matches = ["Psh", "Pun","Kick"]
        matches2 = ["Pr"]
        if any(x in df.at[i, 0] for x in matches):
            ##if df.at[i, 0] == 0:
            # cv2.putText(frame, 'p', (int(df.at[i, 'xmin']), int(df.at[i, 'ymin']) + 20), font,
            #             fontScale, color, thickness, cv2.LINE_4)
            #print("inside write")

            c1 = (df.at[i, 1], df.at[i, 2])
            c2 = (df.at[i, 1]+df.at[i, 3], df.at[i, 2]+df.at[i, 4])
            c3 = (df.at[i, 1]-5, df.at[i, 2]-5)
            cv2.putText(frame, "Violence", c1, font, fontScale, (0, 0, 255), thickness, cv2.LINE_4)
            cv2.rectangle(frame, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)
        elif any(x in df.at[i, 0] for x in matches2):
            c1 = (df.at[i, 1], df.at[i, 2])
            c2 = (df.at[i, 1] + df.at[i, 3], df.at[i, 2] + df.at[i, 4])
            cv2.rectangle(frame, c1, c2, (255, 0, 0), thickness=5, lineType=cv2.LINE_AA)
    result.write(frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(2) & 0xFF == ord('s'):
        break
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#  with io.open(file_name, 'rb') as image_file:
#         #content = image_file.read()
