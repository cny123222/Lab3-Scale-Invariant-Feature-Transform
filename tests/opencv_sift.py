import cv2

gray = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
ret = cv2.drawKeypoints(gray, kp, None)
kp, dst = sift.compute(gray, kp)

cv2.imshow("SIFT Keypoints", ret)
cv2.waitKey(0)
cv2.destroyAllWindows()