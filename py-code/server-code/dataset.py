import tensorflow as tf
import numpy as np
import os
import functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing


class Dataset:

    def __init__(self, DEBUG_MODE, img_dir, label_dir, test_dir, validation_split_ratio, batch_size, random_sized_crops_min,
                 input_size, augment_color, num_parallel_calls=8, predict_on_different_orientations=True):
        self.DEBUG_MODE = DEBUG_MODE
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.test_dir = test_dir
        self.validation_split_ratio = validation_split_ratio
        self.batch_size = batch_size
        self.random_sized_crops_min = random_sized_crops_min
        self.input_size = input_size
        self.augment_color = augment_color
        self.num_parallel_calls = num_parallel_calls
        self.predict_on_different_orientations = predict_on_different_orientations

        self.x_train_filenames = None
        self.x_val_filenames = None
        self.y_train_filenames = None
        self.y_val_filenames = None
        self.x_test_filenames = None

        self.num_train_examples = None
        self.num_val_examples = None
        self.num_test_examples = None

        self._set_dataset_filenames()

    def _set_dataset_filenames(self):
        train_filenames = os.listdir(self.img_dir)
        x_train_filenames = [os.path.join(self.img_dir, filename) for filename in train_filenames]
        y_train_filenames = [os.path.join(self.label_dir, filename) for filename in train_filenames]
        test_filenames = os.listdir(self.test_dir)
        self.x_test_filenames = [os.path.join(self.test_dir, filename) for filename in test_filenames]

        self.x_train_filenames, self.x_val_filenames, self.y_train_filenames, self.y_val_filenames = \
            train_test_split(x_train_filenames,
                             y_train_filenames,
                             test_size=self.validation_split_ratio,
                             random_state=7139)

        self.num_train_examples = len(self.x_train_filenames)
        self.num_val_examples = len(self.x_val_filenames)
        self.num_test_examples = len(self.x_test_filenames)

        print("Number of training examples: {}".format(self.num_train_examples))
        print("Number of validation examples: {}".format(self.num_val_examples))
        print("Number of test examples: {}".format(self.num_test_examples))

        if self.DEBUG_MODE:
            display_num = 2
            r_choices = np.random.choice(self.num_train_examples, display_num)
            plt.figure(figsize=(10, 10))

            for i in range(0, display_num * 2, 2):
                img_num = r_choices[i // 2]
                x_pathname = x_train_filenames[img_num]
                y_pathname = y_train_filenames[img_num]

                plt.subplot(display_num, 2, i + 1)
                plt.imshow(mpimg.imread(x_pathname), cmap="gray")
                plt.subplot(display_num, 2, i + 2)
                plt.imshow(mpimg.imread(y_pathname), cmap="gray")

            plt.suptitle("Examples of Images and their Segmentations")
            plt.show()

    def _process_pathnames(self, fname, label_path):
        img_str = tf.read_file(fname)
        img = tf.image.decode_png(img_str, channels=3)
        label_img_str = tf.read_file(label_path)
        label_img = tf.image.decode_png(label_img_str, channels=1)
        return img, label_img

    def _internal_get_dataset(self, x, y, preprocessing_function, shuffle, repeat, is_batched_first):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=self.num_parallel_calls)

        if not is_batched_first:
            dataset = dataset.map(preprocessing_function, num_parallel_calls=self.num_parallel_calls)

        if shuffle:
            dataset = dataset.shuffle(self.batch_size*50, seed=111, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size)

        if is_batched_first:
            dataset = dataset.map(preprocessing_function, num_parallel_calls=self.num_parallel_calls)

        if repeat:
            dataset = dataset.repeat()

        return dataset

    def get_datasets(self):

        train_config = {
            'mode': 'training',
            'random_sized_crops_min': self.random_sized_crops_min,
            'input_size': self.input_size,
            'change_color': self.augment_color
        }

        validation_config = {
            'mode': 'validation',
            'random_sized_crops_min': self.random_sized_crops_min,
            'input_size': self.input_size,
            'change_color': False
        }

        train_preprocessing_fn = functools.partial(Preprocessing.augment, **train_config)
        validation_preprocessing_fn = functools.partial(Preprocessing.augment, **validation_config)

        if self.predict_on_different_orientations:
            test_config = {
                'input_size': self.input_size
            }
            test_preprocessing_fn = functools.partial(Preprocessing.test_augment, **test_config)
            is_batched_first = True
        else:
            test_config = {
                'mode': 'test',
                'random_sized_crops_min': self.random_sized_crops_min,
                'input_size': self.input_size,
                'change_color': False
            }
            test_preprocessing_fn = functools.partial(Preprocessing.augment, **test_config)
            is_batched_first = False

        train_dataset = self._internal_get_dataset(x=self.x_train_filenames,
                                                   y=self.y_train_filenames,
                                                   preprocessing_function=train_preprocessing_fn,
                                                   shuffle=True,
                                                   repeat=True,
                                                   is_batched_first=False)

        validation_dataset = self._internal_get_dataset(x=self.x_val_filenames,
                                                        y=self.y_val_filenames,
                                                        preprocessing_function=validation_preprocessing_fn,
                                                        shuffle=False,
                                                        repeat=True,
                                                        is_batched_first=False)

        test_dataset = self._internal_get_dataset(x=self.x_test_filenames,
                                                  y=self.x_test_filenames,
                                                  preprocessing_function=test_preprocessing_fn,
                                                  shuffle=False,
                                                  repeat=False,
                                                  is_batched_first=is_batched_first)

        return train_dataset, validation_dataset, test_dataset
