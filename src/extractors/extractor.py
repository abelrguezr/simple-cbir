import numpy as np
import utils.data_utils as utils
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class Extractor:
    def __init__(self, extractor, extractor_type, raw_data_path, processed_data_path = None, n_clusters=10, kmeans = None, process_data = False):
        self.processed_data_path = processed_data_path
        self.extractor = extractor
        self.extractor_type = extractor_type
        self.raw_data_path = raw_data_path
        self.train_data = utils.load_pickle(raw_data_path+'/x_train.pkl')
        self.index = np.array(range(len(self.train_data)))
        self.kmeans = kmeans

        if kmeans == None:
            self.kmeans = self._cluster_images(n_clusters)

        if process_data:
            self.train_data_proc = self._process_data()

        self.train_data_proc = utils.load_pickle(processed_data_path+ '/' +self.extractor_type+'/train_data.pkl')
        

    def _cluster_images(self, n_clusters):
        print("Clustering data with KMeans into " + str(n_clusters) + ' clusters.')
        kmeans = KMeans(n_clusters, random_state=0).fit(self.train_data_proc)
        print("Done running KMeans.")
        utils.save_pickle('../model/kmeans_' + self.extractor_type + '.pkl', kmeans)   

        return kmeans

    def _process_data(self):
        phases=['train','test']
        for phase in phases:
            data = utils.load_pickle(self.raw_data_path+'/x_' + phase +'.pkl')
            feat = utils.apply_func_to_data(self.extract_features, data)
            utils.save_pickle(self.processed_data_path + self.extractor_type +'/' + phase + '_data.pkl', feat)   

    def extract_features(self, data):
        raise NotImplementedError

    def get_knns(self, img):
        # im = im.ravel()
        # Predict label to see in what cluster is assigned
        im = self.extract_features(img)
        label = self.kmeans.predict([im])

        # Create a kNN learner for neighbor searches.
        knn = NearestNeighbors(n_neighbors=4, n_jobs=-2)

        # Fit the learner only with the cluster class data to reduce computations
        knn.fit(self.train_data_proc[self.kmeans.labels_ == label])

        # Aux array with the original indexes in the train data of the same class cluster data
        original_index = self.index[self.kmeans.labels_ == label]

        # Compute 4NN for the given image
        dist, nn = knn.kneighbors([im], 4)

        # Extract the original images from the train_dataset
        nns = self.train_data[original_index[nn[0]]]

        return dist, nns