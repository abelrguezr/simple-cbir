import utils.data_utils
import numpy as np
import cv2
from .extractor import Extractor



class Histogram(Extractor):
    def __init__(self, raw_data_path, process_data, processed_data_path = None, n_clusters=10, kmeans = None):
        self.extractor = self._create_rectangular_masks()
        super().__init__(self.extractor, 'hist',raw_data_path, processed_data_path, n_clusters, kmeans, process_data)

    def _create_rectangular_masks(self):
        (h, w) = (32, 32)
        (cX, cY) = (16, 16)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

        masks = []

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            masks.append(cornerMask)

        return masks

    def extract_features(self, img):
        features = np.array([])
        masks = self.extractor
        for cornerMask in masks:
            hist = self._bgr_hist(img, cornerMask)
            features = np.hstack([features, hist]) if features.size else hist
        return features.ravel()

    def _bgr_hist(self, img, mask):
        result = np.array([])
        for i in range(0, 3):
            hist = cv2.calcHist([img], [i], mask, [32], [0, 256])
            result = np.vstack([result, hist]) if result.size else hist
        return result
