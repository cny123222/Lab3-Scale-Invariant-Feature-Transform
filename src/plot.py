import random
import cv2
import numpy as np
from copy import deepcopy
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Literal, Optional
from sift import calc_grad, calc_main_dir, SIFT

# 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10


def generate_random_colors(num_colors, alpha=1.0):
    """
    生成随机颜色
    """
    colors = []
    for _ in range(num_colors):
        h = random.random()  # 色相随机
        s = random.uniform(0.7, 1.0)  # 饱和度高
        v = random.uniform(0.5, 0.9)  # 亮度适中
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b, alpha))
    return colors


def gray_to_rgba(gray_value, alpha=0.5):
    """
    灰度值转RGBA
    """
    grayscale = gray_value / 255.0
    return (grayscale, grayscale, grayscale, alpha)


def plot_arrow(ax, x, y, angle, magnitude, **kwargs):
    """
    绘制箭头
    """
    ax.arrow(x, y, magnitude * np.cos(angle), magnitude * np.sin(angle), **kwargs)


def find_corners(
        img: np.ndarray # 彩色图片
):
    """
    输出角点坐标
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    return corners.astype(np.int32)


def plot_local_grad(
        img: np.ndarray,  # 彩色图片
        coord: Tuple[int, int],
        type: Literal['figure', 'hist'] = 'figure'
):
    """
    局部梯度可视化
    """
    y, x = coord
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = (img.shape[0] + img.shape[1]) / 700
    m = int(1.5 * sigma)
    magnitudes, angles = calc_grad(img)

    # 绘制直方图（极坐标）
    if type == 'hist':
        _, hist = calc_main_dir(magnitudes, angles, (x, y), m)
        bin_edges = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
        hist = np.concatenate((hist[18:], hist[:18]))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)
        bars = ax.bar(bin_edges, hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
        plt.savefig("figures/local_hist.png", dpi=300)
        plt.show()
    
    # 绘制局部梯度示意图
    elif type == 'figure':
        main_dir, _ = calc_main_dir(magnitudes, angles, (x, y), m)

        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if x - 2 * m <= i < x + 2 * m and y - 2 * m <= j < y + 2 * m:
                    ax.text(j + 0.5, i + 0.5, str(img[i, j]), ha='center', va='center', fontdict={"weight": 'bold', "size": 8})
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(img[i, j])))
                    plot_arrow(ax, j + 0.5, i + 0.5, angles[i, j], magnitudes[i, j] * 0.008, color=(1, 0, 0, 0.9), linewidth=1, head_width=0.2, head_length=0.2)

        ax.add_patch(patches.Rectangle((y - m, x - m), 2 * m, 2 * m, linewidth=2, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((y, x), 1, 1, linewidth=0, edgecolor='none', facecolor=(0, 1, 0, 0.7)))
        plot_arrow(ax, y + 0.5, x + 0.5, main_dir, 6, color=(0, 0, 1, 1), linewidth=2, head_width=0.2, head_length=0.2)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.grid(True)
        plt.savefig("figures/local_grad.png", dpi=300)
        plt.show()


def plot_global_grad(
        img: np.ndarray  # 彩色图片
):
    """
    全局梯度可视化
    """
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
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("figures/global_grad.png", dpi=300)
    plt.show()


def plot_neighbour_grad(
        img: np.ndarray,
        coord: Tuple[int, int],
        magnify: Optional[int] = None
):
    """
    邻域梯度可视化
    """
    y, x = coord
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = (img.shape[0] + img.shape[1]) / 700
    m = int(1.5 * sigma)
    n = int(sigma * 5)
    magnitudes, angles = calc_grad(img)
    
    main_dir, _ = calc_main_dir(magnitudes, angles, (x, y), m)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if x - 3 * n <= i < x + 3 * n and y - 3 * n <= j < y + 3 * n:
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0.05, edgecolor='black', facecolor=gray_to_rgba(img[i, j]))
                ax.add_patch(rect)

    plot_arrow(ax, y + 0.5, x + 0.5, main_dir, 20, color=(1, 0, 0, 1), linewidth=3, head_width=0.3, head_length=0.3)
    plot_arrow(ax, y + 0.5, x + 0.5, main_dir + np.pi / 2, 20, color=(0, 0, 1, 1), linewidth=3, head_width=0.3, head_length=0.3)

    colors = generate_random_colors(16)
    cnt = 0
    for off_i in range(-2, 2):
            for off_j in range(-2, 2):
                cnt += 1

                for i in range(off_i * n, off_i * n + n):
                    for j in range(off_j * n, off_j * n + n):
                        ori_i = x + int(round(i * np.cos(main_dir) - j * np.sin(main_dir)))
                        ori_j = y + int(round(i * np.sin(main_dir) + j * np.cos(main_dir)))

                        plot_arrow(ax, ori_j + 0.5, ori_i + 0.5, angles[ori_i, ori_j], magnitudes[ori_i, ori_j] * 0.005, color=colors[cnt - 1], head_width=0.15, head_length=0.15, linewidth=1.5)
                        
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # 是否局部放大
    if magnify:
        plt.xlim(y - magnify * n, y + magnify * n)
        plt.ylim(x - magnify * n, x + magnify * n)
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("figures/neighbour_grad_magn.png", dpi=300)
    plt.show()


def plot_similarity(
        dst1: np.ndarray,
        dst2: np.ndarray,
):
    """
    描述子相似度可视化
    """
    similarity_matrix = np.dot(dst1, dst2.T)
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='inferno', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("figures/similarity5.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    image1 = cv2.imread("target.jpg")
    # print(find_corners(image1))
    # plot_local_grad(image1, (135, 390), type='figure')
    # plot_local_grad(image1, (135, 390), type='hist')
    # plot_global_grad(image1)
    plot_neighbour_grad(image1, (135, 390), magnify=None)

    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(cv2.imread("dataset/5.jpg"), cv2.COLOR_BGR2GRAY)
    # sift = SIFT()
    # kp1, dst1 = sift.detectAndCompute(image1, corner_num=100)
    # kp2, dst2 = sift.detectAndCompute(image2, corner_num=100)
    # plot_similarity(dst1, dst2)
