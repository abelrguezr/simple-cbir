import utils.data_utils
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from .extractor import Extractor

class CNN(Extractor):
    def __init__(self, raw_data_path, process_data, processed_data_path = None, n_clusters=10, kmeans = None):
        self.extractor = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
        super().__init__(self.extractor, 'cnn', raw_data_path, processed_data_path, n_clusters, kmeans, process_data)


    def extract_features(self, im):
        im = np.expand_dims(im, axis=0)
        vgg16_features = self.extractor.predict(im)
        vgg16_features_np = np.array(vgg16_features)
        return vgg16_features_np.flatten()
