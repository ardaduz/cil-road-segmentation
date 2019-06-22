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


def create_callbacks():
    callbacks = []

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
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

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                               patience=15,
                                                               verbose=1,
                                                               restore_best_weights=True)

    reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                         factor=0.9,
                                                                         patience=10,
                                                                         min_delta=0.03,
                                                                         min_lr=5e-5,
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

    logdir = os.path.join("../runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(logdir)

    zipdir = os.path.join(logdir, "code.zip")
    zippedfiles = zipfile.ZipFile(zipdir, 'w', zipfile.ZIP_DEFLATED)
    zip_code('code/', zippedfiles)
    zippedfiles.close()

    ### CONFIG ###
    DEBUG_MODE = False

    img_dir = '../../competition-data/training/images'
    label_dir = '../../competition-data/training/groundtruth'
    test_dir = '../../competition-data/test'

    validation_split_ratio = 0.2

    random_sized_crops_min = 360  # randomly crops random sized patch, this is resized to 304 later (adds scale augmentation, only training!)
    augment_color = True  # applies slight random hue, contrast, brightness change only on training data

    input_size = 256

    learning_rate = 1e-3
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

    if DEBUG_MODE:
        dummy_dataset, _, _ = dataset.get_datasets()

        data_aug_iter = dummy_dataset.make_one_shot_iterator()
        next_element = data_aug_iter.get_next()
        with tf.Session() as sess:
            batch_of_imgs, label = sess.run(next_element)

            # Running next element in our graph will produce a batch of images
            plt.figure(figsize=(10, 10))
            img = batch_of_imgs[0]

            plt.subplot(1, 2, 1)
            plt.imshow(img)

            plt.subplot(1, 2, 2)
            plt.imshow(label[0, :, :, 0], cmap='gray')
            plt.show()

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
    model = BaselineModel(input_shape=(input_size, input_size, 3), optimizer=optimizer)
    model = model.get_compiled_model()

    callbacks = create_callbacks()

    model.fit(x=train_dataset,
              steps_per_epoch=5*int(np.ceil(dataset.num_train_examples / float(batch_size))),
              epochs=epochs,
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
