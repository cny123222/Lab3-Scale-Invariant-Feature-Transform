import copy
import cv2
import numpy as np


class SIFT:

    def __init__(self, alpha=700, beta=30):
        self.alpha = alpha
        self.beta = beta

    def detect_and_compute(
            self, 
            img_path: str
    ):
        """
        关键点提取及描述子计算
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 超参数计算
        sigma = (img.shape[0] + img.shape[1]) / self.alpha
        radius = int(sigma * self.beta)

        m = int(1.5 * sigma)
        n = radius

        # 关键点提取
        corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10).astype(np.int32)

        # 梯度计算
        magnitudes, angles = self._calc_grad(img)

        kps = []
        dsts = []

        # 逐个处理关键点
        for corner in corners:

            y, x = corner.ravel()

            # 边界检查
            if not m + 1 <= corner[0][1] <= img.shape[0] - m - 1 or not m + 1 <= corner[0][0] <= img.shape[1] - m - 1:
                continue
            kps.append(cv2.KeyPoint(float(y), float(x), 1))

            # 确定主方向
            neighbor_magnitudes = magnitudes[x - m:x + m, y - m:y + m].flatten()
            neighbor_angles = angles[x - m:x + m, y - m:y + m].flatten()
            hist, bin_edges = np.histogram(neighbor_angles, bins=36, range=(-np.pi, np.pi), weights=neighbor_magnitudes)
            main_dir = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2

            # 描述子计算
            dst = []

            for offset_x in range(-2, 2):
                for offset_y in range(-2, 2):
                    near_magnitudes = []
                    near_angles = []

                    for i in range(offset_x * n, offset_x * n + n):
                        for j in range(offset_y * n, offset_y * n + n):

                            # 坐标变换
                            ori_x = x + int(round(i * np.cos(main_dir) - j * np.sin(main_dir)))
                            ori_y = y + int(round(i * np.sin(main_dir) + j * np.cos(main_dir)))
                            
                            try:
                                near_magnitudes.append(magnitudes[ori_x, ori_y])
                                near_angles.append(angles[ori_x, ori_y])
                            except IndexError:
                                pass  # 忽略越界像素
                    
                    # 角度变换及归一化
                    near_angles = np.array(near_angles) - main_dir
                    near_angles[near_angles < -np.pi] += 2 * np.pi
                    near_angles[near_angles > np.pi] -= 2 * np.pi

                    # 更新描述子
                    hist, _ = np.histogram(near_angles, bins=8, range=(-np.pi, np.pi), weights=near_magnitudes)
                    dst.extend(hist)

            # 描述子归一化
            dst = np.array(dst)
            dst = dst / np.linalg.norm(dst)
            dsts.append(dst)

        kps = np.array(kps)
        dst = np.array(dsts, dtype=np.float32)
        return kps, dst

    def _calc_grad(
            self, 
            img: np.ndarray
    ):
        img = img.astype(np.int32)
        img_border = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)  # 边界填充
        grad_x = img_border[2:, 1:-1] - img_border[:-2, 1:-1]  # x方向梯度
        grad_y = img_border[1:-1, 2:] - img_border[1:-1, :-2]  # y方向梯度
        magnitudes = np.sqrt(grad_x ** 2 + grad_y ** 2)  # 梯度幅值
        angles = np.arctan2(grad_y, grad_x)  # 梯度方向
        return magnitudes, angles