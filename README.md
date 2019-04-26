# CIL-Road Segmentation Project
Computational Intelligence Lab (Road Segmentation Project) - ETH Zürich Spring 2019

Group Members:
* Marilou Beyeler
* Arda Duzceker
* Jonas Hein
* Juan Lopez Fernandez

## Data Collection and Preprocessing

### Extra Data Collection
Using data-processing/collect_data.py script and data-processing/input_cities.csv parameters, 
extra data from 5 large US (1000 each) cities has been collected until now. 
The Google Drive link to the data:
https://drive.google.com/file/d/1RX-Ctq6ULIj9nihJEEUX35sQKGJrZvXG/view?usp=sharing

## Models

### baseline-cnn
Code taken from https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb .
I just made minor changes in the data preprocessing and postprocessing steps:
- Adjusted the input pipeline to work with our data
- Added random 90-degree rotations to the preprocessing pipeline
- Added root mean squared error as a metric for selecting the best model/checkpoint
- Added early stopping
- Resizing input images and predictions to match the required shape
- Added the provided code to mask our predictions for submission
- Probably some minor things that I forgot

### baseline-graph-cut
TODO
