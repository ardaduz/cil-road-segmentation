# Road Segmentation from Aerial Images
Computational Intelligence Lab (Road Segmentation Project) - ETH Zürich Spring 2019

Group Name: Uncalibrated

Group Members:
* Arda Duzceker
* Jonas Hein
* Juan Lopez Fernandez
* Marilou Beyeler

### Dependencies
* python 3.7.x
* tensorflow 1.13.1
* opencv 3.4.x
* numpy
* scikit-learn
* pillow
* matplotlib
* tqdm
* scipy
* pandas
* imageio
* pymaxflow (for graph-cut baseline, can be installed using the provided master branch folder)

### competition-data
Please download the competition data from Kaggle and place its contents into ./competition-data folder. 
The empty folder structure is given for guidance purposes.

### additional-data
Running additional-data/collect_data.py using additional-data/input_cities.csv parameters for number of images 
and city bounding boxes, extra data can be gathered. The Google Drive link to already gathered Google Maps data:
https://drive.google.com/open?id=1aXjASwNVKF6bc4CWXcC_TU6Q2BfUr5Lp

### google-maps-data
If you want to reproduce the results, please download the additional data that we used using the link above,
unzip the file and replace ./google-maps-data empty folder in the root directory which is placed only for guidance purposes.

### baseline-graph-cut
Please install PyMaxflow library following the instructions in ./baseline-graph-cut/PyMaxflow-master/README.rst

Then running ./baseline-graph-cut/run.py without any arguments will reproduce our Graph-Cut baseline predictions.

### baseline-cnn
Code adapted from 
https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb

Simply run whole IPython notebook ./baseline-cnn/baseline-cnn.ipynb from scratch to train our U-Net baseline 
with competition data only and reproduce our results.

### ensemble
This folder contains training codes of individual models, ImageNet pretrained weights that we used, a log directory
and prediction codes for ensemble prediction after training all the models.

##### runs
This log folder is initially empty, but please do not remove it. When you run one the trainings, a log folder named 
with the run time and date is created, all tensorboard materials, current situation of code and best checkpoint is 
saved here for later use.

##### training_mobilenetv2_based_model_with_single_spatial_pyramid
The folder contains MobilenetV2 Encoder with Spatial Pyramid Pooling and Upsampling Decoder model. Running training.py
will start the training procedure described in the report. The required best checkpoint of this model is saved 
as ./ensemble/runs/yyyymmdd-hhmmss/mobilenetv2_based_model_best_checkpoint.hdf5

##### training_xception_based_model_with_two_spatial_pyramids
The folder contains Xception Encoder with Spatial Pyramid Pooling and Upsampling Decoder model. Running training.py
will start the training procedure described in the report. The required best checkpoint of this model is saved 
as ./ensemble/runs/yyyymmdd-hhmmss/xception_based_model_with_two_pyramids_best_checkpoint.hdf5

##### training_xception_based_model_with_many_spatial_pyramids
The folder contains Xception Encoder with Many Spatial Pyramid Pooling Blocks and Upsampling Decoder model. 
Running training.py will start the training procedure described in the report. The required best checkpoint of 
this model is saved as ./ensemble/runs/yyyymmdd-hhmmss/xception_based_model_with_many_pyramids_best_checkpoint.hdf5

##### prediction
The folder contains the code necessary to perform Ensemble and Equivariant Prediction section described in the report.
After training all(!) the models individually, please locate their best checkpoints in their respective log folders. Then,
move/copy all of three checkpoints into ./ensemble/prediction/ folder and run ./ensemble/prediction/prediction.py

For convenience reasons, we also provide our best checkpoints with which we acquired our competition score:
https://drive.google.com/open?id=1aXjASwNVKF6bc4CWXcC_TU6Q2BfUr5Lp

The ./ensemble/prediction/results folder will be created automatically and 
it will contain predictions for test images. Also, the submission file named 
"mobilenet_xceptiontwo_xceptionmany_rotation_flip_ensemble_majority_sigmoid.csv" will appear in this directory.

### NOTE
If something here is not clear, we are sorry, please contact us in such a situation.
