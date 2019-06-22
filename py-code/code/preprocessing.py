import tensorflow as tf


class Preprocessing:

    @staticmethod
    def crop_and_resize_img(img, label, random_sized_crops_min, input_size):
        raw_size = tf.shape(img)[1]

        crop_size = tf.random_uniform([], minval=random_sized_crops_min, maxval=raw_size-1, dtype=tf.int32, seed=420)
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

        is_training = mode == 'training'

        if is_training:
            img, label = Preprocessing.crop_and_resize_img(img=img,
                                                           label=label,
                                                           random_sized_crops_min=random_sized_crops_min,
                                                           input_size=input_size)

            img, label = Preprocessing.flip_img(img, label)
            img, label = Preprocessing.rot_img(img, label)

            if change_color:
                img = tf.image.random_hue(img, max_delta=0.1, seed=420)
                img = tf.image.random_contrast(img, lower=0.0, upper=0.1, seed=420)
                img = tf.image.random_brightness(img, max_delta=0.2, seed=420)

        else:  # test or validation, don't do anything, just resize to the desired input size
            img = tf.image.resize(img, size=[input_size, input_size])
            label = tf.image.resize(label, size=[input_size, input_size])

        img = tf.image.per_image_standardization(img)
        label = label / 255.0

        return img, label
