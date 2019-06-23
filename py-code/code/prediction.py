import glob
import os
import re
from PIL import Image
import imageio
import tensorflow as tf
from tensorflow.python.keras import models
import numpy as np
import matplotlib.image as mpimg
import pandas as pd

from dataset import Dataset
from model import *


class Prediction:
    def __init__(self, test_dataset, test_filenames, model, logdir, model_path):
        self.test_dataset = test_dataset
        self.test_filenames = test_filenames
        self.model = model
        self.logdir = logdir
        self.model_path = model_path

        ### Mask to submission
        self.foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

        self.prediction_dir = os.path.join(self.logdir, "predictions")
        os.mkdir(self.prediction_dir)

    def predict(self):
        test_iterator = self.test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()
        count = 0

        sess = tf.keras.backend.get_session()
        try:
            while True:
                batch_of_imgs, label = sess.run(test_next_element)
                predicted_labels = self.model.predict(batch_of_imgs)

                # rescale images from [0, 1] to [0, 255]
                predicted_labels = predicted_labels * 255.0

                for i in range(len(predicted_labels)):
                    pred = Image.fromarray(predicted_labels[i, :, :, 0], 'F').resize((608, 608)).convert('L')
                    test_filename = self.test_filenames[count]
                    index = test_filename.rfind("/") + 1
                    test_filename = test_filename[index:]
                    imageio.imwrite(os.path.join(self.prediction_dir, test_filename), pred)
                    count += 1

        except tf.errors.OutOfRangeError:
            pass

    def submit(self):
        submission_filename = model_path.replace("weights", "submission", 1).replace(".hdf5", ".csv", 1)
        image_filenames = [os.path.join(self.prediction_dir, filename) for filename in os.listdir(self.prediction_dir)]
        self.masks_to_submission(submission_filename, *image_filenames)

        print("Validating submission file:", submission_filename)
        df = pd.read_csv(submission_filename)

        print('Shape of csv:', df.shape)
        assert df.shape == (135736, 2), "Invalid number of rows or columns in submission file!"
        assert df['id'].unique().size == 135736, "Column 'id' should contain 135736 unique values!"

        meanPred = df['prediction'].mean()
        print("Mean prediction: {:.3f}".format(meanPred))
        assert meanPred > 0.05 and meanPred < 0.3, "Very unlikely mean prediction!"

        print("Submission file looks OKAY!")

    # assign a label to a patch
    def patch_to_label(self, patch):
        df = np.mean(patch)
        if df > self.foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(self, image_filename):
        """Reads a single image and outputs the strings that should go into the submission file"""
        img_number = int(re.search(r"\d+(?=\.png$)", image_filename).group(0))
        im = mpimg.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = self.patch_to_label(patch)
                yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

    def masks_to_submission(self, submission_filename, *image_filenames):
        """Converts images into a submission file"""
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s) for s in self.mask_to_submission_strings(fn))


if __name__ == "__main__":
    ### SET THIS !!! ###
    logdir = "/home/ardaduz/ETH/CIL/project/cil-road-segmentation/py-code/runs/20190623-021923"

    img_dir = '../../competition-data/training/images'
    label_dir = '../../competition-data/training/groundtruth'
    test_dir = '../../competition-data/test'

    validation_split_ratio = 0.2

    random_sized_crops_min = 384  # randomly crops random sized patch, this is resized to 304 later (adds scale augmentation, only training!)
    augment_color = True  # applies slight random hue, contrast, brightness change only on training data

    input_size = 384

    learning_rate = 1e-3
    batch_size = 4
    epochs = 100

    dataset = Dataset(DEBUG_MODE=False,
                      img_dir=img_dir,
                      label_dir=label_dir,
                      test_dir=test_dir,
                      validation_split_ratio=validation_split_ratio,
                      batch_size=batch_size,
                      random_sized_crops_min=random_sized_crops_min,
                      input_size=input_size,
                      augment_color=augment_color,
                      num_parallel_calls=8)

    train_dataset, validation_dataset, test_dataset = dataset.get_datasets()

    cp_dir = os.path.join(logdir, 'model*')
    model_path = sorted(glob.glob(cp_dir), reverse=True)[0]
    model = models.load_model(model_path, custom_objects={'root_mean_squared_error': LossesMetrics.root_mean_squared_error,
                                                          'bce_dice_loss': LossesMetrics.bce_dice_loss,
                                                          'dice_loss': LossesMetrics.dice_loss})

    predictor = Prediction(test_dataset=test_dataset,
                           test_filenames=dataset.x_test_filenames,
                           model=model,
                           logdir=logdir,
                           model_path=model_path)

    predictor.predict()
    predictor.submit()
