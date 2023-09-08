import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()
    output = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (15, 15), 0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    canny = cv.Canny(th3, 100, 200)

    edges, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(output, edges, -1, (0, 255, 0), 3)

    cv.imshow("otsu", th3)
    cv.imshow("webcam", output)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()