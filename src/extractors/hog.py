import utils.data_utils
import numpy as np
import cv2
from .extractor import Extractor


class HOG(Extractor):
    def __init__(self, raw_data_path, process_data, processed_data_path = None, n_clusters=10, kmeans = None):
        
        winSize = (32, 32)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        self.extractor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) 

        super().__init__(self.extractor, 'hog', raw_data_path, processed_data_path, n_clusters, kmeans, process_data)


    def extract_features(self, im):
        return self.extractor.compute(im).T.ravel()

