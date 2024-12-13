import random
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Literal
from sift import calc_grad, calc_main_dir

def generate_random_colors(num_colors):
    return [(random.random(), random.random(), random.random(), 1) for _ in range(num_colors)]

def gray_to_rgba(gray_value, alpha=0.5):
    grayscale = gray_value / 255.0
    return (grayscale, grayscale, grayscale, alpha)


def plot_arrow(ax, x, y, angle, magnitude, **kwargs):
    ax.arrow(x, y, magnitude * np.cos(angle), magnitude * np.sin(angle), **kwargs)


def find_corners(
        img: np.ndarray
):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    return corners.astype(np.int32)

def plot_local_grad(
        img: np.ndarray,
        coord: Tuple[int, int],
        type: Literal['figure', 'hist'] = 'figure'
):
    y, x = coord
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = (img.shape[0] + img.shape[1]) / 700
    m = int(1.5 * sigma)
    magnitudes, angles = calc_grad(img)

    if type == 'hist':
        _, hist = calc_main_dir(magnitudes, angles, (x, y), m)
        bin_edges = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
        hist = np.concatenate((hist[18:], hist[:18]))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)
        bars = ax.bar(bin_edges, hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
        plt.show()
    
    elif type == 'figure':
        main_dir, _ = calc_main_dir(magnitudes, angles, (x, y), m)

        fig, ax = plt.subplots()
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if x - 2 * m <= i < x + 2 * m and y - 2 * m <= j < y + 2 * m:
                    ax.text(j + 0.5, i + 0.5, str(img[i, j]), ha='center', va='center', fontsize=8)
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(img[i, j]))
                    ax.add_patch(rect)
                    plot_arrow(ax, j + 0.5, i + 0.5, angles[i, j], magnitudes[i, j] * 0.008, color=(0, 0, 1, 0.8), linewidth=1, head_width=0.1, head_length=0.1)

        rect = patches.Rectangle((y - m, x - m), 2 * m, 2 * m, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((y, x), 1, 1, linewidth=0, edgecolor='none', facecolor=(0, 1, 0, 0.8))
        ax.add_patch(rect)

        plot_arrow(ax, y + 0.5, x + 0.5, main_dir, 5, color=(1, 0, 0, 1), linewidth=2, head_width=0.2, head_length=0.2)
                        
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def plot_global_grad(
        img: np.ndarray
):
    img_copy = deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = (img.shape[0] + img.shape[1]) / 700
    m = int(1.5 * sigma)

    corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    corners = corners.astype(np.int32)
    magnitudes, angles = calc_grad(img)

    for corner in corners:
        y, x = corner.ravel()
        if not m + 1 <= x <= img.shape[0] - m - 1 or not m + 1 <= y <= img.shape[1] - m - 1:
            continue
        main_dir, _ = calc_main_dir(magnitudes, angles, (x, y), m)

        cv2.circle(img_copy, (y, x), 1, (0, 0, 255), -1)
        end_x = int(x + 20 * np.cos(main_dir))
        end_y = int(y + 20 * np.sin(main_dir))
        cv2.arrowedLine(img_copy, (y, x), (end_y, end_x), (0, 0, 255), 1)

    grad_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(grad_img)
    plt.axis("off")
    plt.show()


def plot_neighbour_grad(
        img: np.ndarray,
        coord: Tuple[int, int]
):
    y, x = coord
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = (img.shape[0] + img.shape[1]) / 700
    m = int(1.5 * sigma)
    n = int(sigma * 25)
    magnitudes, angles = calc_grad(img)
    
    main_dir, _ = calc_main_dir(magnitudes, angles, (x, y), m)

    fig, ax = plt.subplots()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if x - 3 * n <= i < x + 3 * n and y - 3 * n <= j < y + 3 * n:
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(img[i, j]))
                ax.add_patch(rect)

    rect = patches.Rectangle((y - m, x - m), 2 * m, 2 * m, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    plot_arrow(ax, y + 0.5, x + 0.5, main_dir, 150, color=(1, 0, 0, 1), linewidth=4, head_width=0.3, head_length=0.3)
    plot_arrow(ax, y + 0.5, x + 0.5, main_dir + np.pi / 2, 150, color=(0, 0, 1, 1), linewidth=4, head_width=0.3, head_length=0.3)

    colors = generate_random_colors(16)
    cnt = 0
    for off_i in range(-2, 2):
            for off_j in range(-2, 2):
                cnt += 1

                for i in range(off_i * n, off_i * n + n):
                    for j in range(off_j * n, off_j * n + n):
                        ori_i = x + int(round(i * np.cos(main_dir) - j * np.sin(main_dir)))
                        ori_j = y + int(round(i * np.sin(main_dir) + j * np.cos(main_dir)))

                        plot_arrow(ax, ori_j + 0.5, ori_i + 0.5, angles[ori_i, ori_j], magnitudes[ori_i, ori_j] * 0.01, color=colors[cnt - 1], head_width=0.1, head_length=0.1)
                        
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_similarity(
        dst1: np.ndarray,
        dst2: np.ndarray,
):
    similarity_matrix = np.dot(dst1, dst2.T)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='inferno', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    image1 = cv2.imread("target.jpg")
    plot_local_grad(image1, (417, 279), type='figure')
    plot_neighbour_grad(image1, (417, 279))
    # print(find_corners(image1))