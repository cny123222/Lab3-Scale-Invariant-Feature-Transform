import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import random

dst_1 = None
dst_2 = None

def generate_random_colors(num_colors, deci=True):
    colors = []
    for _ in range(num_colors):
        if not deci:
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        else:
            colors.append((random.random(), random.random(), random.random(), 1))
    return colors

def plot_arrow(ax, x, y, angle, magnitude, color=(1, 0, 0, 0.8), linewidth=1):
    ax.arrow(x, y, magnitude * np.cos(angle), magnitude * np.sin(angle), head_width=0.2, head_length=0.2, color=color, linewidth=linewidth)

def gray_to_rgba(gray_value):
    grayscale = gray_value / 255.0
    return (grayscale, grayscale, grayscale, 0.5)

def compute_dst(img_path, a, b):

    image = cv2.imread(img_path)
    # print(image.shape)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    sigma = (gray.shape[0] + gray.shape[1]) / 700
    radius = int(sigma * 30)
    print(sigma, radius)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = corners.astype(np.int32)

    gray = gray.astype(np.int32)
    gray_border = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    grad_x = gray_border[2:, 1:-1] - gray_border[:-2, 1:-1]
    grad_y = gray_border[1:-1, 2:] - gray_border[1:-1, :-2]
    magnitudes = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angles = np.arctan2(grad_y, grad_x)

    dst = []
    new_corners = []

    image_copy = copy.deepcopy(image)

    for corner in corners:

        y, x = corner.ravel()
        # print(x, y)

        condition = x == b and y == a
        # condition = True
        # condition = False

        cv2.circle(image_copy, (y, x), 1, (0, 0, 255), -1)

        m = int(1.5 * sigma)
        if not m + 1 <= x <= gray.shape[0] - m - 1 or not m + 1 <= y <= gray.shape[1] - m - 1:
            continue
        new_corners.append(cv2.KeyPoint(float(y), float(x), 1))
        dst_c = []

        # if x == a and y == b:
        if False:

            fig, ax = plt.subplots()

            for i in range(0, gray.shape[0]):
                for j in range(0, gray.shape[1]):
                    if x - m <= i < x + m and y - m <= j < y + m:
                        ax.text(i + 0.5, j + 0.5, str(gray[i, j]), ha='center', va='center', fontsize=4)
                        rect = patches.Rectangle((i, j), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(gray[i, j]))
                        ax.add_patch(rect)
                        plot_arrow(ax, i + 0.5, j + 0.5, angles[i, j], magnitudes[i, j] * 0.01)
            
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0, edgecolor='none', facecolor=(0, 1, 0, 0.8))
            ax.add_patch(rect)
                    
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.show()
        
        neighbor_mag = magnitudes[x - m:x + m, y - m:y + m].flatten()
        neighbor_angles = angles[x - m:x + m, y - m:y + m].flatten()
        hist, bin_edges = np.histogram(neighbor_angles, bins=36, range=(-np.pi, np.pi), weights=neighbor_mag)
        # if x == a and y == b:
        #     plt.hist(neighbor_angles, bins=36, range=(-np.pi, np.pi), weights=neighbor_mag)
        #     plt.show()
        main_dir = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2

        # print(main_dir)

        arrow_length = 20
        end_x = int(x + arrow_length * np.cos(main_dir))
        end_y = int(y + arrow_length * np.sin(main_dir))
        cv2.arrowedLine(image_copy, (y, x), (end_y, end_x), (0, 0, 255), 1)

        colors = generate_random_colors(16, deci=True)
        cnt = 0

        if condition:
            fig, ax = plt.subplots()
            for i in range(0, gray.shape[0]):
                for j in range(0, gray.shape[1]):
                    if x - 2 * m <= i < x + 2 * m and y - 2 * m <= j < y + 2 * m:
                        ax.text(i + 0.5, j + 0.5, str(gray[i, j]), ha='center', va='center', fontsize=4)
                        rect = patches.Rectangle((i, j), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(gray[i, j]))
                        ax.add_patch(rect)

            plot_arrow(ax, x + 0.5, y + 0.5, main_dir, 10, color=(1, 0, 0, 1), linewidth=2)
            plot_arrow(ax, x + 0.5, y + 0.5, main_dir + np.pi / 2, 10, color=(0, 0, 1, 1), linewidth=2)

        for off_i in range(-2, 2):
            for off_j in range(-2, 2):
                cnt += 1
                near_angles = []
                weights = []

                n = radius

                for i in range(off_i * n, off_i * n + n):
                    for j in range(off_j * n, off_j * n + n):
                        ori_i = x + int(round(i * np.cos(main_dir) - j * np.sin(main_dir)))
                        ori_j = y + int(round(i * np.sin(main_dir) + j * np.cos(main_dir)))

                        if condition:
                            plot_arrow(ax, ori_i + 0.5, ori_j + 0.5, angles[ori_i, ori_j], magnitudes[ori_i, ori_j] * 0.01, color=colors[cnt - 1])

                        # cv2.circle(image_copy, (ori_i, ori_j), 1, colors[cnt - 1], -1)
                    
                        try:
                            near_angles.append(angles[ori_i, ori_j])
                            weights.append(magnitudes[ori_i, ori_j])
                        except IndexError:
                            pass
                # plt.hist(near_angles, bins=8, range=(-np.pi, np.pi))
                # plt.show()

                near_angles = np.array(near_angles) - main_dir
                near_angles[near_angles < -np.pi] += 2 * np.pi
                near_angles[near_angles > np.pi] -= 2 * np.pi

                hist, bin_edges = np.histogram(near_angles, bins=8, range=(-np.pi, np.pi), weights=weights)
                dst_c.extend(hist)

                # if condition:
                #     print(cnt, hist)

        if condition:
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.show()

        if condition:
        # if True:
            cv2.imshow('Good Features to Track Corners', image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # break
                
        dst_c = np.array(dst_c)
        dst_c = dst_c / np.linalg.norm(dst_c)
        dst.append(dst_c)

        if condition and a == 135:
            global dst_1
            dst_1 = dst_c
        elif condition and a == 68:
            global dst_2
            dst_2 = dst_c

    # cv2.imshow('Good Features to Track Corners', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    new_corners = np.array(new_corners)
    dst = np.array(dst, dtype=np.float32)

    return new_corners, dst


image1 = cv2.imread("target.jpg")
image2 = cv2.imread("dataset/3.jpg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

kp1, dst1 = compute_dst("target.jpg", a=135, b=390)
kp2, dst2 = compute_dst("dataset/3.jpg", a=68, b=158)

# plt.plot(np.arange(128), dst_1, color='r')
# plt.plot(np.arange(128), dst_2, color='b')
# plt.show()

colors = generate_random_colors(len(kp1), deci=False) 

h1, w1 = gray1.shape
h2, w2 = gray2.shape
img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
img_matches[:h1, :w1] = image1
img_matches[:h2, w1:] = image2

# res = dst1 @ dst2.T
# res_max = res.max(axis=1)
# con = kp2[res.argmax(axis=1)]
# for i in range(len(kp1)):
#     if res_max[i] > 0.9:
#         color = colors[i]
#         cv2.circle(image1, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), 8, color, -1)
#         cv2.circle(image2, (int(con[i].pt[0]), int(con[i].pt[1])), 8, color, -1)
#         cv2.line(img_matches, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), (int(con[i].pt[0]) + w1, int(con[i].pt[1])), color, 2)

# cv2.imshow("SIFT Matches with Filtering", img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


bf = cv2.BFMatcher()
matches = bf.knnMatch(dst1, dst2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

match_img = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("SIFT Matches with Filtering", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()