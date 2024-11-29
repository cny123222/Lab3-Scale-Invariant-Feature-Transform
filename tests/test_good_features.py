import cv2
import numpy as np

image = cv2.imread("target.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

corners = np.int64(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

cv2.imshow('Good Features to Track Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()