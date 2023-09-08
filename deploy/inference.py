from segmentation import YOLOsegmentation 
import cv2 as cv
import numpy as np

ys = YOLOsegmentation("best.pt")

img = cv.imread("robocon_0005.jpg")
scale = 800/img.shape[1]
img = cv.resize(img, None, fx=scale, fy=scale)

bboxes, classes, segmentation, scores = ys.detect(img)

for bbox, class_id, seg, score in zip(bboxes, classes, segmentation, scores):
    (x1, y1, x2, y2) = bbox
    cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv.polylines(img, [seg], True, (255, 255, 255), 1)
    points = []
    for i in seg:
        points.append([i[0], i[1]])
    points.sort(key=lambda x: x[1])
    px = int(sum([x[0] for x in points[:4]])/4)
    py = int(sum([x[1] for x in points[:4]])/4)
    cv.circle(img, (px,py), 5, (0, 255, 255), -1)
    cv.putText(img, "pole "+str(score), (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

cv.imshow('A', img)
cv.waitKey(0)




