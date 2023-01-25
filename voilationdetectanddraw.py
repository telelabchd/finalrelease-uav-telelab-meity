#def violation_detect(huma_centroids):
#,max_distance_between_people):
import cv2
from itertools import combinations
from scipy.spatial import distance
low_risk_dist = 250
high_risk_dist = 200
thickness = 5
radius = 10
def socialdistance(img, human_centroids):
    if not human_centroids:
        return [-2]
    n = 0
    violation_pts_x = []
    violation_pts_y = []
    # tuple(map(int, tup))
    already_red = dict()
    points = [tuple(map(int, (x, y))) for (x, y) in human_centroids]
    for j in points:
        already_red[j] = 0
    points_combo = list(combinations(points, 2))

    for i in points_combo:
        dist = distance.euclidean(i[0], i[1])
        cntr1, cntr2 = i[0], i[1]
        print("this is xy1: ", cntr1, "this is xy2: ", cntr2)
        if dist > high_risk_dist and dist < low_risk_dist:
            color = (0, 255, 255)

            label = "Low Risk "
            cv2.line(img, cntr1, cntr2, color, thickness)
            if already_red[cntr1] == 0:
                cv2.circle(img, cntr1, radius, color, -1)
            if already_red[cntr2] == 0:
                cv2.circle(img, cntr2, radius, color, -1)
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
            for cntr in [cntr1, cntr2]:
                if already_red[cntr] == 0:
                    c1, c2 = i[0], i[1]
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 255], thickness=tf,
                                lineType=cv2.LINE_AA)

        elif dist < high_risk_dist:
            color = (0, 0, 255)
            label = "High Risk"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            c1, c2 = i[0], i[1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
