import cv2
import numpy as np
from matplotlib import pyplot as plt

image1 = cv2.imread("target.jpg")
image2 = cv2.imread("dataset/3.jpg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 创建白色背景
background = np.ones_like(match_img, dtype=np.uint8) * 255

# 将匹配图像叠加到白色背景
matches_img_white_bg = np.where(match_img > 0, match_img, background)

plt.imshow(cv2.cvtColor(matches_img_white_bg, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()