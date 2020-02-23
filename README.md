# simple-cbir
Implementation of a simple system for Content-based image retrieval (CBIR) using Keras and OpenCV on the CIFAR10 dataset.
As an experiment, three methods have been implemented to define the features to compare, histograms, a HOG extractor, and a pre-trained VGG16 for feature extraction. 
The features are then clustered with k-means to determine the closest match in a query.

## Usage

See src/main.py for all the options

```
pip install -r requirements.txt
cd src
python main.py --extractor_type <'Hist|'HOG'|'CNN'> --test_image_index <img_index>
```

## Results

### RGB Histogram

The use of RGB histograms in general does not provide satisfactory results to obtain images of similar content, since it only considers color. In cases where there are repeated or distorted images is possible to obtain good results, but this is not commonly the case, because only images with similar color are considered, which do not have to represent the target object.

![image](https://user-images.githubusercontent.com/34161053/72837860-80603500-3c8f-11ea-9fd6-66f337360c2d.png)

### Histogram of Oriented Gradients (HOG)

The performance of this technique is better than in the previous case, and behaves well in cases where the orientation of the object is similar across the images. Color variation can be observed in the images found, so this approach is robust in situations where color is not relevant for grouping image content.

![image](https://user-images.githubusercontent.com/34161053/72837810-6b83a180-3c8f-11ea-9cd9-6103ff2dd43a.png)

### VGG16

As expected, the state-of-the-art CNN works better (what a surprise, right? :scream:)

![image](https://user-images.githubusercontent.com/34161053/72837776-5870d180-3c8f-11ea-82eb-35c7104e974b.png)