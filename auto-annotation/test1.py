import cv2 as cv

image = cv.imread('ring.jpg')
half = cv.resize(image, (0, 0), fx = 0.5, fy = 0.5)
gray = cv.cvtColor(half, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blur, 100, 200)
contours, hierarchy= cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(half, contours, -1, (0, 255, 0), 1)
cv.imshow('Ring', half)
cv.waitKey(0)