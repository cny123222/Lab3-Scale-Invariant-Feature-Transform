import cv2
import numpy as np

image1 = cv2.imread("target.jpg")
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(gray1, 2, 3, 0.1)

dst = cv2.dilate(dst, None)

image1[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
