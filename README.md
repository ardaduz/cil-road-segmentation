# CIL-Road Segmentation Project
Computational Intelligence Lab (Road Segmentation Project) - ETH Zürich Spring 2019

### Group Members
* Marilou Beyeler
* Arda Duzceker
* Jonas Hein
* Juan Lopez Fernandez

### Project Report
The final report can be found [TODO](TODO)

# Setup

### Dependencies
TODO

* Python 3.5.3+
* Tensorflow 1.1.0
* Numpy
* scikit-learn
* scikit-image
* Pillow
* Matplotlib
* Tqdm
* scipy

### Folder structure
TODO describe the folder structure and where to put the data, additional data, model-weight files, ...

## Extra Data Collection
Using additional-data/collect_data.py script and additional-data/input_cities.csv parameters, 
extra data (1000 image pairs each) from 5 large US cities has been collected. 
The Google Drive link to the data:
https://drive.google.com/drive/folders/1aXjASwNVKF6bc4CWXcC_TU6Q2BfUr5Lp

# Models

### baseline-cnn
Code based on https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb .
To reproduce the baseline, simply run the jupyter notebook located at baseline-cnn/baseline-cnn.ipynb

### baseline-graph-cut
TODO

## Ensemble
TODO
first train the three ensemble models, then do this and that to get the final predictions.

### Mobilenetv2 based model with single spatial pyramid
TODO

### Xception based model with two spatial pyramids
TODO

### Xception based model with many spatial pyramids
TODO

# Results
TODO - do we add the results somewhere?

