import click
import numpy as np
import utils.data_utils as utils
import cv2
import os
from extractors.main import get_extractor
from keras.datasets import cifar10


@click.command()
@click.option('--extractor_type',
              type=click.Choice(['Hist', 'HOG','CNN'], case_sensitive=False), help='Extractor to use for CBIR.')
@click.option('--test_image_index', type=click.IntRange(0, 9999, clamp=True),show_default=True, help='Index (0-9999) of image on the test dataset to query the CBIR.')
@click.option('--download', is_flag=True, help='Set the flag for downloading the CIFAR10 dataset. If downloaded already this flag is not needed')
@click.option('--raw_data_path', type=click.Path(exists=True), help='Path to download the data when --download is set or for loading the data otherwise.')
@click.option('--processed_data_path', type=click.Path(exists=True), help='Path to save the processed data the data when --process_data is set or for loading the data otherwise.')
@click.option('--n_clusters', default=10, show_default=True, help='Number of kmeans clusters to use.')
@click.option('--kmeans_data_path', type=click.Path(exists=True), help='Path to load the model with kmeans if set. If not, the model will be saved on ../model')
@click.option('--process_data', is_flag=True, help='Set the flag for processing the CIFAR10 dataset.')



def main(extractor_type,test_image_index, download, raw_data_path, processed_data_path, n_clusters, kmeans_data_path, process_data):

    raw_data_path=os.path.abspath(raw_data_path)
    processed_data_path=os.path.abspath(processed_data_path)

    if(download):
        (x_train,y_train),(x_test,y_test)=cifar10.load_data()

        utils.save_pickle(raw_data_path+'/x_train.pkl',x_train)
        utils.save_pickle(raw_data_path+'/y_train.pkl',y_train)
        utils.save_pickle(raw_data_path+'/x_test.pkl',x_test)
        utils.save_pickle(raw_data_path+'/y_test.pkl',y_test)

    kmeans = None
    if (kmeans_data_path):
        kmeans_data_path=os.path.abspath(kmeans_data_path)
        kmeans = utils.load_pickle(kmeans_data_path)

    extractor = get_extractor(extractor_type, raw_data_path, processed_data_path, n_clusters, kmeans, process_data)
    test_data = utils.load_pickle(raw_data_path+'/x_test.pkl')
    im = test_data[test_image_index]
    _,nns = extractor.get_knns(im)
    display_nn(im,nns)


def display_nn(im, nns):
    im = cv2.resize(im, (0, 0), fx=4, fy=4)

    cv2.imshow('Original image', im)

    result = np.array([])
    for nn in nns:
        result = np.hstack([result, nn]) if result.size else nn

    result = cv2.resize(result, (0, 0), fx=4, fy=4)

    cv2.imshow('NNs', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()    
