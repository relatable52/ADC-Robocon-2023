import cv2 as cv
import numpy as np
import copy

def detectPole(image):
    upper = np.array([38, 255, 255])
    lower = np.array([25, 80, 20])
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,8))

    scale = 800/image.shape[0]
    small = cv.resize(image, (0, 0), fx = scale, fy = scale)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV_FULL)
    mask = cv.inRange(hsv, lower, upper)
    erode = cv.erode(mask, kernel=kernel, iterations=5)
    dilate = cv.dilate(erode, kernel=kernel, iterations=6)
    blur = cv.GaussianBlur(dilate, (35, 35), 0)
    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    edges = cv.Canny(otsu, 100, 200)

    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rect = [None]*len(contours)
    for i, c in enumerate(contours):
        rect[i] = cv.minAreaRect(c)
        (x,y), (width, height), angle = rect[i]
        ratio = min(width, height)/max(width,height)
        if ratio<0.3:
            # box = cv.boxPoints(rect[i])
            # box = np.intp(box)
            # cv.drawContours(small, [box], 0, (0, 255, 0))
            tip = list(list(j[0]) for j in contours[i])
            tip.sort(key=lambda x: x[1])
            x = int(sum(j[0] for j in tip[0:10])/10)
            y = int(sum(j[1] for j in tip[0:10])/10)
            cv.circle(small, (x, y), 5, (255, 200, 0), -1)
            cv.drawContours(small, contours, i,  (0, 255, 0), 1)
            

    return small, otsu  

video = cv.VideoCapture('pole5.mp4')
while True:
    ret, frame = video.read()

    output, hsv = detectPole(frame)
    cv.imshow('A', output)
    cv.imshow('B', hsv)

    if cv.waitKey(20) & 0xff == ord('d'):
        break

video.release()
cv.destroyAllWindows()


