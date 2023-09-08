import cv2 as cv
import numpy as np


upper = np.array([180, 90, 255])
lower = np.array([120, 10, 10])
kernel = kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

image = cv.imread('pole.jpg')
scale = 480/image.shape[0]
small = cv.resize(image, (0,0), fx=scale, fy=scale)
hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV_FULL)
mask = cv.inRange(hsv, lower, upper)
erode = cv.erode(src=mask, kernel=kernel, iterations=10)
dilate = cv.dilate(src=erode, kernel=kernel, iterations=10)
blur = cv.GaussianBlur(dilate, (15, 15), 0)
ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
canny = cv.Canny(otsu, 100, 200)
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

minRect = [None]*len(contours)
for i, c in enumerate(contours):
    minRect[i] = cv.minAreaRect(c)
    (x, y), (width, height), angle = minRect[i]
    ratio = min(width, height)/ max(width, height)
    if ratio < 0.3:
      box = cv.boxPoints(minRect[i])
      box = np.intp(box)
      cv.drawContours(small, [box], 0, (0, 255, 0))
    
cv.imshow('A', small)
cv.imshow('B', otsu)
cv.imshow('C', erode)
cv.waitKey(0)