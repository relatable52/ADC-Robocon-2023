import cv2 as cv
import numpy as np

upper = np.array([35, 85, 255])
lower = np.array([28, 76, 0])

def savelabel(path, save):
    image = cv.imread(path)
    scale = 640/image.shape[0]
    image = cv.resize(image, (0,0), fx=scale, fy=scale)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.inRange(image, lower, upper)
    image = cv.erode(image, (5, 5), iterations=5)
    image = cv.dilate(image, (5, 5), iterations=5)
    cv.rectangle(image, (0, 0), (1183, 640), (0, 0, 0), 5)
    image = cv.GaussianBlur(image, (15, 15), 0)
    ret, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #cv.imshow('A', image)
    cv.imwrite(save, image)

for i in range(1, 301):
    order = str("{:04d}".format(i))
    path = r'gen_label\robocon_' + order + '.jpg'
    save = r'label\robocon_' + order + '.jpg'
    print(path)
    savelabel(path, save)

