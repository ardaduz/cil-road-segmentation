import glob
import os
import re
import tensorflow as tf
from tensorflow.python.keras import models, layers
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from dataset import Dataset
from model import *


class Prediction:
    def __init__(self, test_dataset, test_filenames, models, logdir, model_path, submission_filename,
                 predict_on_different_orientations=True, DEBUG_MODE=False):
        self.test_dataset = test_dataset
        self.test_filenames = test_filenames
        self.models = models
        self.logdir = logdir
        self.model_path = model_path
        self.predict_on_different_orientations = predict_on_different_orientations
        self.DEBUG_MODE = DEBUG_MODE
        self.submission_filename = submission_filename

        ### Mask to submission
        self.foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

        self.prediction_dir = os.path.join(self.logdir, "predictions")
        os.mkdir(self.prediction_dir)

    def predict(self, batch_size):
        test_iterator = self.test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()
        count = 0

        if self.predict_on_different_orientations:
            batch_size = batch_size * 8

        sess = tf.keras.backend.get_session()
        try:
            while True:
                batch_of_imgs, _ = sess.run(test_next_element)
                sum_predicted_labels = np.expand_dims(np.zeros((1, batch_of_imgs.shape[1], batch_of_imgs.shape[2])), axis=-1)

                for current_model in self.models:
                    predicted_labels = current_model.predict(batch_of_imgs)

                    if self.DEBUG_MODE:
                        plt.figure(figsize=(10, 10))
                        for i in range(0, batch_size):
                            img = batch_of_imgs[i]

                            plt.subplot(batch_size, 2, i * 2 + 1)
                            plt.imshow(img)

                            plt.subplot(batch_size, 2, i * 2 + 2)
                            plt.imshow(predicted_labels[i, :, :, 0], cmap='gray')
                        plt.show()

                    if self.predict_on_different_orientations:
                        predicted_labels = self.postprocess(predicted_labels)

                    if self.DEBUG_MODE:
                        plt.figure()
                        plt.imshow(predicted_labels[0, :, :, 0], cmap='gray')
                        plt.show()

                    sum_predicted_labels += predicted_labels

                predicted_labels = sum_predicted_labels / float(len(self.models))
                predicted_labels = (predicted_labels > 0.5).astype(np.float64)

                # rescale images from [0, 1] to [0, 255]
                predicted_labels = predicted_labels * 255.0
                predicted_labels = np.maximum(predicted_labels, 0.0)
                predicted_labels = np.minimum(predicted_labels, 255.0)

                for i in range(predicted_labels.shape[0]):
                    test_filename = self.test_filenames[count]
                    index = test_filename.rfind("/") + 1
                    test_filename = test_filename[index:]
                    save_filename = os.path.join(self.prediction_dir, test_filename)
                    pred = predicted_labels[i, :, :, 0].astype(np.uint8)
                    pred = cv2.resize(pred, (608, 608), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(save_filename, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    count += 1

        except tf.errors.OutOfRangeError:
            pass

    def postprocess(self, batch):
        def _postprocess_combine_flips(sample_batch):
            """Combine original predictions with flipped predictions by averaging them"""
            # splitted[0] = original, splitted[1] = flipped
            splitted = np.split(sample_batch, 2)
            flip_reversed = splitted[1]
            flip_reversed = flip_reversed[:, :, ::-1, :]
            combined = np.mean(np.stack([splitted[0], flip_reversed], axis=0), axis=0)
            return combined

        def _postprocess_combine_rot90(sample_batch):
            """Combine original prediction with rotated predictions by averaging them"""
            # splitted[0] = 0°, splitted[1] = 90°, splitted[2] = 180°, splitted[3] = 270°
            splitted = np.split(sample_batch, 4)
            rot_reversed = [np.rot90(splitted[i], 4 - i, axes=(1, 2)) for i in range(4)]
            combined = np.mean(np.stack(rot_reversed, axis=0), axis=0)
            return combined

        # combine flipped predictions with originals
        batch = _postprocess_combine_rot90(batch)

        # combine flipped predictions with originals
        batch = _postprocess_combine_flips(batch)

        return batch

    def submit(self):
        submission_filename = self.submission_filename
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
    logdir = "." #/home/ardaduz/ETH/CIL/project/cil-road-segmentation/py-code/runs/20190625-014252"

    submission_filename = "mobilenet_xception_xceptionfullconnected_rotation_flip_ensemble_majority_sigmoid.csv"

    img_dir = '../../competition-data/training/images'
    label_dir = '../../competition-data/training/groundtruth'
    test_dir = '../../competition-data/test'

    validation_split_ratio = 0.2

    random_sized_crops_min = 608  # randomly crops random sized patch, this is resized to 304 later (adds scale augmentation, only training!)
    augment_color = True  # applies slight random hue, contrast, brightness change only on training data

    input_size = 384

    learning_rate = 1e-3
    batch_size = 1
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

    # MODEL 1
    model_path = "model_epoch199_rmse0.0397.hdf5"
    model = models.load_model(model_path, custom_objects={'root_mean_squared_error': LossesMetrics.root_mean_squared_error,
                                                          'bce_dice_loss': LossesMetrics.bce_dice_loss,
                                                          'dice_loss': LossesMetrics.dice_loss})

    model.save_weights(os.path.join(logdir, "best_model_weights.hdf5"))
    xception_model = XceptionSpatialPyramid(input_shape=(input_size, input_size, 3), optimizer=None)
    xception_model = xception_model.get_model()
    xception_model.load_weights(os.path.join(logdir, "best_model_weights.hdf5"))

    # MODEL 2
    model_path = "model_epoch172_rmse0.0456.hdf5"
    model = models.load_model(model_path, custom_objects={'root_mean_squared_error': LossesMetrics.root_mean_squared_error,
                                                          'bce_dice_loss': LossesMetrics.bce_dice_loss,
                                                          'dice_loss': LossesMetrics.dice_loss})

    model.save_weights(os.path.join(logdir, "best_model_weights.hdf5"))
    mobilenet_model = MobilenetV2SpatialPyramid(input_shape=(input_size, input_size, 3), optimizer=None)
    mobilenet_model = mobilenet_model.get_model()
    mobilenet_model.load_weights(os.path.join(logdir, "best_model_weights.hdf5"))

    # MODEL 3
    model_path = "best_model.hdf5"
    model = models.load_model(model_path, custom_objects={'root_mean_squared_error': LossesMetrics.root_mean_squared_error,
                                                          'bce_dice_loss': LossesMetrics.bce_dice_loss,
                                                          'dice_loss': LossesMetrics.dice_loss})

    model.save_weights(os.path.join(logdir, "best_model_weights.hdf5"))
    xceptionarda_model = XceptionSpatialPyramidArda(input_shape=(input_size, input_size, 3), optimizer=None)
    xceptionarda_model = xceptionarda_model.get_model()
    xceptionarda_model.load_weights(os.path.join(logdir, "best_model_weights.hdf5"))

    predictor = Prediction(test_dataset=test_dataset,
                           test_filenames=dataset.x_test_filenames,
                           models=[xceptionarda_model, xception_model, mobilenet_model],
                           logdir=logdir,
                           model_path=model_path,
                           submission_filename=submission_filename)

    predictor.predict(batch_size=batch_size)
    predictor.submit()
