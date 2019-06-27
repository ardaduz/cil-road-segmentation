import os
import glob
import zipfile
import functools
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.image as mpimg
import pandas as pd
import imageio
from PIL import Image

import tensorflow as tf

from dataset import Dataset
from prediction import Prediction
from model import *

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)


class TensorBoardWithLR(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, write_graph):
        super().__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'learning_rate': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def create_callbacks():
    callbacks = []

    tensorboard_callback = TensorBoardWithLR(
        log_dir=logdir,
        write_graph=True
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(logdir, 'model_epoch{epoch:03d}_rmse{val_root_mean_squared_error:.4f}.hdf5'),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        period=1,
        save_weights_only=False)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=20,
                                                               verbose=1,
                                                               restore_best_weights=True)

    reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                         factor=0.5,
                                                                         patience=10,
                                                                         min_delta=0.00001,
                                                                         min_lr=1e-6,
                                                                         cooldown=5,
                                                                         verbose=1)

    callbacks.append(tensorboard_callback)
    callbacks.append(model_checkpoint_callback)
    callbacks.append(early_stopping_callback)
    callbacks.append(reduce_lr_on_plateau_callback)

    return callbacks


def zip_code(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == "__main__":

    tf.random.set_random_seed(324)
    np.random.seed(324)

    logdir = os.path.join("../runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(logdir)

    zipdir = os.path.join(logdir, "code.zip")
    zippedfiles = zipfile.ZipFile(zipdir, 'w', zipfile.ZIP_DEFLATED)
    zip_code('./', zippedfiles)
    zippedfiles.close()

    ### CONFIG ###
    DEBUG_MODE = True
    img_dir = '../../competition-data/training/images'
    label_dir = '../../competition-data/training/groundtruth'
    test_dir = '../../competition-data/test'

    google_img_dir = '../../google-maps-data/images'
    google_label_dir = '../../google-maps-data/groundtruth'
    google_test_dir = '../../competition-data/test'

    validation_split_ratio = 0.15

    random_sized_crops_min = 336  # randomly crops random sized patch, this is resized later (adds scale augmentation, only training!)
    augment_color = True  # applies slight random hue, contrast, brightness change only on training data

    input_size = 256

    batch_size = 8
    epochs = 100

    ### START ###
    dataset = Dataset(DEBUG_MODE=DEBUG_MODE,
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

    google_dataset = Dataset(DEBUG_MODE=DEBUG_MODE,
                             img_dir=google_img_dir,
                             label_dir=google_label_dir,
                             test_dir=google_test_dir,
                             validation_split_ratio=validation_split_ratio,
                             batch_size=batch_size,
                             random_sized_crops_min=random_sized_crops_min,
                             input_size=input_size,
                             augment_color=augment_color,
                             num_parallel_calls=8)

    google_train_dataset, google_validation_dataset, google_test_dataset = google_dataset.get_datasets()

    if DEBUG_MODE:
        dummy_dataset, _, _ = google_dataset.get_datasets()

        data_aug_iter = dummy_dataset.make_one_shot_iterator()
        next_element = data_aug_iter.get_next()
        with tf.Session() as sess:
            batch_of_imgs, label = sess.run(next_element)

            plt.figure(figsize=(10, 10))
            for i in range(0, batch_size):
                # Running next element in our graph will produce a batch of images
                img = batch_of_imgs[i]

                plt.subplot(batch_size, 2, i * 2 + 1)
                plt.imshow(img)

                plt.subplot(batch_size, 2, i * 2 + 2)
                plt.imshow(label[i, :, :, 0], cmap='gray')
            plt.show()

    callbacks = create_callbacks()

    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    base_model = MobilenetV2SpatialPyramid(input_shape=(input_size, input_size, 3), optimizer=optimizer)
    model = base_model.get_model()
    model.compile(optimizer=optimizer,
                  loss=LossesMetrics.dice_loss,
                  metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])
    print(model.summary())
    model.fit(x=google_train_dataset,
              steps_per_epoch=int(np.ceil(google_dataset.num_train_examples / float(batch_size))),
              epochs=15,
              validation_data=google_validation_dataset,
              validation_steps=int(np.ceil(google_dataset.num_val_examples / float(batch_size))),
              callbacks=callbacks)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=optimizer,
                  loss=LossesMetrics.dice_loss,
                  metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])
    print(model.summary())
    model.fit(x=google_train_dataset,
              steps_per_epoch=int(np.ceil(google_dataset.num_train_examples / float(batch_size))),
              epochs=300,
              validation_data=google_validation_dataset,
              validation_steps=int(np.ceil(google_dataset.num_val_examples / float(batch_size))),
              callbacks=callbacks)

    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=optimizer,
                  loss=LossesMetrics.dice_loss,
                  metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])
    print(model.summary())
    model.fit(x=train_dataset,
              steps_per_epoch=5 * int(np.ceil(dataset.num_train_examples / float(batch_size))),
              epochs=300,
              validation_data=validation_dataset,
              validation_steps=int(np.ceil(dataset.num_val_examples / float(batch_size))),
              callbacks=callbacks)

    validation_iterator = validation_dataset.make_one_shot_iterator()
    validation_next_element = validation_iterator.get_next()
    plt.figure(figsize=(10, 20))
    for i in range(5):
        batch_of_imgs, label = tf.keras.backend.get_session().run(validation_next_element)
        img = batch_of_imgs[0]
        predicted_label = model.predict(batch_of_imgs)[0]

        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(img)
        plt.title("Input image")

        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(label[0, :, :, 0], cmap='gray')
        plt.title("Actual Mask")
        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(predicted_label[:, :, 0], cmap='gray')
        plt.title("Predicted Mask")
    plt.suptitle("Qualitative Validation Dataset - Input Image, Label, and Prediction")
    plt.show()

    predictor = Prediction(test_dataset=test_dataset,
                           test_filenames=dataset.x_test_filenames,
                           model=model,
                           logdir=logdir)

    predictor.predict()
    predictor.submit()
