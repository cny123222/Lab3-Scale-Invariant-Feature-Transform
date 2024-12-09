import os
import sys
import warnings
import argparse
import cv2
import matplotlib.pyplot as plt
from sift import SIFT
from utils import get_image_paths


def parse_args(args):
    """
    读取命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="dataset", help="path to the input image or folder of input images")
    parser.add_argument("--target-dir", type=str, default="target.jpg", help="path to the target image")
    parser.add_argument("--output-dir", type=str, default="result", help="path to folder of output images")
    parser.add_argument("--output-type", type=str, default="png", help="layout of output images")

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)

    sift = SIFT()
    image1 = cv2.imread(args.target_dir)
    kps1, dst1 = sift.detect_and_compute(image1)

    img_paths = get_image_paths(args.image_dir)
    for img_path in img_paths:
        image2 = cv2.imread(img_path)
        kps2, dst2 = sift.detect_and_compute(image2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dst1, dst2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        match_img = cv2.drawMatches(image1, kps1, image2, kps2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        original_filename = os.path.splitext(os.path.basename(img_path))[0]
        save_pth = os.path.join(args.output_dir, f"{original_filename}-mySift.{args.output_type}")
        if os.path.exists(save_pth):
            warnings.warn(f"File '{save_pth}' already exists. Existing file will be overwritten.", UserWarning)

        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        plt.imshow(match_img)
        plt.axis("off")
        plt.savefig(save_pth, bbox_inches='tight', pad_inches=0, dpi=300)

    print(f"{len(img_paths)} figures successfully saved to {args.output_dir}")

if __name__ == '__main__':
    main(sys.argv[1:])