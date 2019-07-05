import tensorflow as tf


class Preprocessing:

    @staticmethod
    def rotate_and_center_crop_img(img, label, raw_size, crop_size):
        crop_size_float = tf.cast(crop_size, dtype=tf.float32)
        raw_size_float = tf.cast(raw_size, dtype=tf.float32)

        x_plus_y = raw_size_float
        x_times_y = (tf.square(raw_size_float) - tf.square(crop_size_float)) / tf.constant(2.0)
        x_minus_y = tf.sqrt(tf.square(raw_size_float) - 4 * x_times_y)
        x = (x_plus_y + x_minus_y) / 2.0
        y = (x_plus_y - x_minus_y) / 2.0

        max_angle = tf.atan(y / x)

        random_angle = tf.random_uniform([], minval=-max_angle, maxval=max_angle, seed=123456)

        img = tf.contrib.image.rotate(images=img,
                                      angles=random_angle,
                                      interpolation='BILINEAR')
        label = tf.contrib.image.rotate(images=label,
                                        angles=random_angle,
                                        interpolation='BILINEAR')

        offset = tf.cast((raw_size - crop_size) / 2, dtype=tf.int32)
        img = tf.image.crop_to_bounding_box(img,
                                            offset_height=offset,
                                            offset_width=offset,
                                            target_height=crop_size,
                                            target_width=crop_size)
        label = tf.image.crop_to_bounding_box(label,
                                              offset_height=offset,
                                              offset_width=offset,
                                              target_height=crop_size,
                                              target_width=crop_size)
        return img, label

    @staticmethod
    def dont_rotate_and_random_crop_img(img, label, raw_size, crop_size):
        maxval = raw_size - crop_size
        offset_height = tf.random_uniform([], minval=0, maxval=maxval, dtype=tf.int32, seed=420)
        offset_width = tf.random_uniform([], minval=0, maxval=maxval, dtype=tf.int32, seed=420)

        img = tf.image.crop_to_bounding_box(img,
                                            offset_height=offset_height,
                                            offset_width=offset_width,
                                            target_height=crop_size,
                                            target_width=crop_size)

        label = tf.image.crop_to_bounding_box(label,
                                              offset_height=offset_height,
                                              offset_width=offset_width,
                                              target_height=crop_size,
                                              target_width=crop_size)

        return img, label

    @staticmethod
    def rotate_crop_resize_img(img, label, random_sized_crops_min, input_size):
        raw_size = tf.shape(img)[1]
        crop_size = tf.random_uniform([], minval=random_sized_crops_min, maxval=raw_size - 1, dtype=tf.int32, seed=420)

        rotate_prob = tf.random_uniform([], 0.0, 1.0, seed=1983)
        img, label = tf.cond(tf.less(rotate_prob, 0.5),
                             lambda: (Preprocessing.rotate_and_center_crop_img(img, label, raw_size, crop_size)),
                             lambda: (img, label))
        img, label = tf.cond(tf.greater_equal(rotate_prob, 0.5),
                             lambda: (Preprocessing.dont_rotate_and_random_crop_img(img, label, raw_size, crop_size)),
                             lambda: (img, label))

        img = tf.image.resize(img, size=[input_size, input_size])
        label = tf.image.resize(label, size=[input_size, input_size])
        return img, label

    @staticmethod
    def flip_img(img, label):
        left_right_flip_prob = tf.random_uniform([], 0.0, 1.0, seed=420)
        img, label = tf.cond(tf.less(left_right_flip_prob, 0.5),
                             lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(label)),
                             lambda: (img, label))

        up_down_flip_prob = tf.random_uniform([], 0.0, 1.0, seed=420)
        img, label = tf.cond(tf.less(up_down_flip_prob, 0.5),
                             lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(label)),
                             lambda: (img, label))
        return img, label

    @staticmethod
    def rot_img(img, label):
        rotate_count = tf.random_uniform([], 0, 4, dtype=tf.dtypes.int32, seed=420)
        img = tf.image.rot90(img, rotate_count)
        label = tf.image.rot90(label, rotate_count)
        return img, label

    @staticmethod
    def augment(img,
                label,
                mode,
                random_sized_crops_min,
                input_size,
                change_color):
        # if mode is test img and label parameters are batches of images, otherwise they contain single image

        is_training = mode == 'training'

        if is_training:
            img, label = Preprocessing.rotate_crop_resize_img(img=img,
                                                              label=label,
                                                              random_sized_crops_min=random_sized_crops_min,
                                                              input_size=input_size)

            img, label = Preprocessing.flip_img(img, label)
            img, label = Preprocessing.rot_img(img, label)

            if change_color:
                img = tf.image.random_hue(img, max_delta=0.1, seed=420)
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2, seed=420)
                img = tf.image.random_brightness(img, max_delta=0.2, seed=420)

                img = tf.minimum(img, 255.0)
                img = tf.maximum(img, 0.0)
        else:  # validation, don't do anything, just resize to the desired input size
            img = tf.image.resize(img, size=[input_size, input_size])
            label = tf.image.resize(label, size=[input_size, input_size])

        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        label = tf.cast(label, dtype=tf.float32)
        label_min = tf.reduce_min(label)
        label_max = tf.reduce_max(label)
        label = tf.cond(tf.equal(label_max, label_min), lambda: label, lambda: (label - label_min) / (label_max - label_min))

        return img, label

    @ staticmethod
    def test_augment(img,
                     label_img,
                     input_size):

        img = tf.image.resize(img, size=[input_size, input_size])
        label_img = tf.image.resize(label_img, size=[input_size, input_size])

        tmp = [img, tf.image.flip_left_right(img)]
        img = tf.concat(tmp, axis=0)
        tmp = [label_img, tf.image.flip_left_right(label_img)]
        label_img = tf.concat(tmp, axis=0)

        # Concatenate all four 90° rotations
        tmp = [tf.image.rot90(img, i) for i in range(4)]
        img = tf.concat(tmp, axis=0)
        tmp = [tf.image.rot90(label_img, i) for i in range(4)]
        label_img = tf.concat(tmp, axis=0)

        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        return img, label_img
