from .cnn import CNN
from .hog import HOG
from .hist import Histogram

def get_extractor(extractor_type, raw_data_path, proc_data_path, n_clusters, kmeans, process_data):

    implemented_extractors = ('Hist', 'HOG', 'CNN')
    assert extractor_type in implemented_extractors


    if extractor_type == 'Hist':
        extractor = Histogram(raw_data_path=raw_data_path,processed_data_path=proc_data_path,n_clusters=n_clusters, kmeans=kmeans, process_data=process_data)

    if extractor_type == 'HOG':
        extractor = HOG(raw_data_path=raw_data_path,processed_data_path=proc_data_path,n_clusters=n_clusters, kmeans=kmeans, process_data=process_data)

    if extractor_type == 'CNN':
        extractor = CNN(raw_data_path=raw_data_path,processed_data_path=proc_data_path,n_clusters=n_clusters, kmeans=kmeans, process_data=process_data)

    return extractor    