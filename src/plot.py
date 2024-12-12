import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple

from sift import SIFT


def gray_to_rgba(gray_value, alpha=0.5):
    grayscale = gray_value / 255.0
    return (grayscale, grayscale, grayscale, alpha)


def plot_arrow(ax, x, y, angle, magnitude, color=(1, 0, 0, 0.8), linewidth=1):
    ax.arrow(x, y, magnitude * np.cos(angle), magnitude * np.sin(angle), head_width=0.2, head_length=0.2, color=color, linewidth=linewidth)


def plot_local_grad(
        image: np.ndarray,
        coord: Tuple[int, int]
):
    pass

def plot_global_grad(
        image: np.ndarray
):
    pass

def plot_neighbour_grad(
        image: np.ndarray,
        coord: Tuple[int, int]
):
    pass

def _plot_gradient(
        sift: SIFT,
        coord: Tuple[int, int]
):
    image = sift.image
    m = sift.m
    x, y = coord

    fig, ax = plt.subplots()
    for i in range(0, sift.image.shape[0]):
        for j in range(0, image.shape[1]):
            if x - 2 * m <= i < x + 2 * m and y - 2 * m <= j < y + 2 * m:
                ax.text(i + 0.5, j + 0.5, str(image[i, j]), ha='center', va='center', fontsize=4)
                rect = patches.Rectangle((i, j), 1, 1, linewidth=0, edgecolor='none', facecolor=gray_to_rgba(image[i, j]))
                ax.add_patch(rect)
                plot_arrow(ax, i + 0.5, j + 0.5, sift.angles[i, j], sift.magnitudes[i, j] * 0.01)

    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, edgecolor='none', facecolor=(0, 1, 0, 0.8))
    ax.add_patch(rect)
                    
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    sift = SIFT()
    image1 = cv2.imread("target.jpg")
    kps1, dst1 = sift.detect_and_compute(image1)

    _plot_gradient(sift, (135, 390))