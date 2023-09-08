import cv2 as cv
import numpy as np

def createBlobDetector():
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 50000

    params.filterByCircularity = True
    params.minCircularity = 0.82

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv.SimpleBlobDetector_create(params)

    return detector

def processImage(image):
    scale = 480/image.shape[0]
    small = cv.resize(image, (0, 0), fx = scale, fy = scale)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    et, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #otsu = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    #otsu = cv.erode(otsu, (10, 10), iterations=2)
    test = np.ones((small.shape[0], small.shape[1]), dtype=np.uint8)
    test = 255*test
    canny = cv.Canny(otsu, 100, 200)
    canny = cv.dilate(canny, (10, 10), iterations=10)

    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(test, contours, -1, (0, 0, 0), -1)
    #cv.drawContours(test, contours, -1, (255, 255, 255), 3)
    #test = cv.erode(test, (1,1), iterations=5)
    
    return test

capture = cv.VideoCapture(1)

detector = createBlobDetector()

while True:
    isTrue, frame = capture.read()
    output = cv.resize(frame, (0, 0), fx = 480/frame.shape[0], fy = 480/frame.shape[0])
    processed = processImage(frame)

    keypoints = detector.detect(processed)
    
    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(output, keypoints, blank, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("A", blobs)
    cv.imshow("B", processed)
    #cv.imshow("C", contours)
    print(keypoints)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

