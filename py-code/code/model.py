from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
import numpy as np


class LossesMetrics:
    # Dice loss is a metric that measures overlap. More info on optimizing for Dice coefficient (our dice loss) can be found in the
    # [paper](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf), where it was introduced. We use dice loss here because
    # it performs better at class imbalanced problems by design. In addition, maximizing the dice coefficient and IoU metrics are the
    # actual objectives and goals of our segmentation task. Using cross entropy is more of a proxy which is easier to maximize.
    # Instead, we maximize our objective directly.

    @staticmethod
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    @staticmethod
    def dice_loss(y_true, y_pred):
        loss = 1 - LossesMetrics.dice_coeff(y_true, y_pred)
        return loss

    # Here, we'll use a specialized loss function that combines binary cross entropy and our dice loss. This is based on
    # [individuals who competed within this competition obtaining better results empirically]
    # (https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199).
    # Try out your own custom losses to measure performance (e.g. bce + log(dice_loss), only bce, etc.)!
    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        loss = LossesMetrics.dice_loss(y_true, y_pred)
        return loss

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        loss = tf.sqrt(losses.mean_squared_error(y_true, y_pred))
        return loss

    @staticmethod
    def jaccard_loss(y_true, y_pred, smooth=100):
        """Jaccard distance for semantic segmentation.
        Also known as the intersection-over-union loss.
        This loss is useful when you have unbalanced numbers of pixels within an image
        because it gives all classes equal weight. However, it is not the defacto
        standard for image segmentation.
        For example, assume you are trying to predict if
        each pixel is cat, dog, or background.
        You have 80% background pixels, 10% dog, and 10% cat.
        If the model predicts 100% background
        should it be be 80% right (as with categorical cross entropy)
        or 30% (with this loss)?
        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 100.
        # Returns
            The Jaccard distance between the two tensors.
        # References
            - [What is a good evaluation measure for semantic segmentation?](
               http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
        """
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
        sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        loss = (1 - jac) * smooth
        return loss

    @staticmethod
    def tversky_loss(y_true, y_pred):
        beta = 0.2
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return numerator / (tf.reduce_sum(denominator) + tf.keras.backend.epsilon())


class BaselineModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.model = None
        self.build_model()

    def conv_bn_relu(self, x, filters, kernel_size=[3, 3], dilation_rate=1):
        encoder = layers.Conv2D(filters, kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_bn_relu(input_tensor, num_filters)
        encoder = self.conv_bn_relu(encoder, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(encoder)
        return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def upsampling_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3))(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3))(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def spatial_pyramid_block(self, spatial_pyramid_input, filters, dilation_rates):
        b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[1, 1], dilation_rate=dilation_rates[0])
        b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[1])
        b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[2])

        b3 = layers.AveragePooling2D(pool_size=[4, 4], padding='valid')(spatial_pyramid_input)
        b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b3 = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(b3)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=filters * 2, kernel_size=[1, 1], dilation_rate=1)
        return spatial_pyramid_output

    def build_model(self):
        dropout_rate = 0.5
        seed = 1234

        inputs = layers.Input(shape=self.input_shape)
        # 256

        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)
        # 128

        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)
        # 64

        encoder1_pool = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder1_pool)
        encoder1 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder1)

        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)
        # 32

        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)
        # 16

        encoder3_pool = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder3_pool)
        encoder3 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder3)

        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)
        # 8

        # spatial pyramids for last encoders
        encoder4_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder4_pool, filters=256, dilation_rates=[1, 2, 3])

        decoder4 = self.upsampling_block(encoder4_pyramid, encoder4, 512)
        decoder4 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(decoder4)
        # 16

        decoder3 = self.upsampling_block(decoder4, encoder3, 256)
        # 32

        decoder2 = self.upsampling_block(decoder3, encoder2, 128)
        decoder2 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(decoder2)
        # 64

        decoder1 = self.upsampling_block(decoder2, encoder1, 64)
        # 128

        decoder0 = self.upsampling_block(decoder1, encoder0, 32)
        # 256

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        self.model = model

    def get_model(self):
        return self.model


class EncoderMLPModel:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape

        self.model = None

        self.build_model()

        self.model.compile(optimizer=optimizer,
                           loss=LossesMetrics.bce_dice_loss,
                           metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])

        print(self.model.summary())

    def convolution_block(self, x, num_filters, kernel_sizes, strides):
        x = layers.Conv2D(num_filters, (kernel_sizes[0], kernel_sizes[0]), strides=(strides[0], strides[0]), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(num_filters, (kernel_sizes[1], kernel_sizes[1]), strides=(strides[1], strides[1]), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        conv1 = self.convolution_block(x=inputs, num_filters=32, kernel_sizes=[5, 3], strides=[2, 1])
        pool1 = layers.pooling.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv1)

        conv2 = self.convolution_block(x=pool1, num_filters=64, kernel_sizes=[3, 3], strides=[1, 1])
        pool2 = layers.pooling.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv2)

        conv3 = self.convolution_block(x=pool2, num_filters=128, kernel_sizes=[3, 3], strides=[1, 1])
        pool3 = layers.pooling.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv3)

        conv4 = self.convolution_block(x=pool3, num_filters=256, kernel_sizes=[3, 3], strides=[1, 1])
        pool4 = layers.pooling.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same')(conv4)

        up1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(pool1)
        up2 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(pool2)
        up3 = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(pool3)
        up4 = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(pool4)

        concat = layers.concatenate([up1, up2, up3, up4])

        intermediate = layers.Dense(units=512)(concat)
        intermediate = layers.BatchNormalization()(intermediate)
        intermediate = layers.Activation('relu')(intermediate)

        intermediate = layers.Conv2DTranspose(filters=256, kernel_size=[2, 2], strides=[2, 2], padding='same')(intermediate)

        outputs = layers.Dense(1, activation='sigmoid')(intermediate)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        self.model = model

    def get_compiled_model(self):
        return self.model


class XceptionSpatialPyramid:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.encoder_spatial_dropout_rate = 0.0
        self.decoder_spatial_dropout_rate = 0.0
        self.l2_regularization = 0.0

        self.model = None
        self.build_model()

    def build_pretrained_xception_model(self, input_layer):
        encoder_model = tf.keras.applications.Xception(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=input_layer,
                                                       input_shape=self.input_shape,
                                                       pooling=None,
                                                       classes=None)

        for layer in encoder_model.layers:
            layer.trainable = False

        encoder1 = encoder_model.get_layer('block2_sepconv2_bn').output
        encoder2 = encoder_model.get_layer('block3_sepconv2_bn').output
        encoder3 = encoder_model.get_layer('block4_sepconv2_bn').output
        encoder4 = encoder_model.get_layer('block13_sepconv2_bn').output
        spatial_pyramid_input = encoder_model.output

        encoder1 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder1)
        encoder2 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder2)
        encoder3 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder3)
        encoder4 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder4)
        spatial_pyramid_input = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(spatial_pyramid_input)

        return encoder1, encoder2, encoder3, encoder4, spatial_pyramid_input

    def conv_bn_relu(self, x, filters, kernel_size, dilation_rate):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def spatial_pyramid_block(self, spatial_pyramid_input, filters, dilation_rates):

        b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[1, 1], dilation_rate=dilation_rates[0])
        b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[1])
        b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[2])

        b3 = layers.AveragePooling2D(pool_size=[4, 4], padding='valid')(spatial_pyramid_input)
        b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b3 = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(b3)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=filters * 2, kernel_size=[1, 1], dilation_rate=1)
        return spatial_pyramid_output

    def decoder_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='valid')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)

        _, concat_tensor_image_size, _, _ = concat_tensor.shape.as_list()
        _, decoder_image_size, _, _ = decoder.shape.as_list()
        if concat_tensor_image_size != decoder_image_size:
            pad_begin = int(np.ceil((decoder_image_size - concat_tensor_image_size) / 2.0))
            pad_end = int(np.floor((decoder_image_size - concat_tensor_image_size) / 2.0))
            concat_tensor = layers.ZeroPadding2D(padding=((pad_begin, pad_end), (pad_begin, pad_end)))(concat_tensor)

        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def upsampling_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(input_tensor)

        _, concat_tensor_image_size, _, _ = concat_tensor.shape.as_list()
        _, decoder_image_size, _, _ = decoder.shape.as_list()
        if concat_tensor_image_size != decoder_image_size:
            pad_begin = int(np.ceil((decoder_image_size - concat_tensor_image_size) / 2.0))
            pad_end = int(np.floor((decoder_image_size - concat_tensor_image_size) / 2.0))
            concat_tensor = layers.ZeroPadding2D(padding=((pad_begin, pad_end), (pad_begin, pad_end)))(concat_tensor)

        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        zero_padded_inputs = layers.ZeroPadding2D(padding=(3, 3))(inputs)

        encoder1, encoder2, encoder3, encoder4, center = self.build_pretrained_xception_model(zero_padded_inputs)

        encoder4_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder4, filters=128, dilation_rates=[1, 3, 5])
        encoder4_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder4_pyramid)

        center_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=center, filters=192, dilation_rates=[1, 2, 3])
        center_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(center_pyramid)

        # without deconv
        decoder4 = self.upsampling_block(input_tensor=center_pyramid, concat_tensor=encoder4_pyramid, num_filters=256, padding='same')
        decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)

        encoder3 = self.conv_bn_relu(x=encoder3, filters=96, kernel_size=[1, 1], dilation_rate=1)
        decoder3 = self.upsampling_block(input_tensor=decoder4, concat_tensor=encoder3, num_filters=192, padding='same')
        decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)

        encoder2 = self.conv_bn_relu(x=encoder2, filters=64, kernel_size=[1, 1], dilation_rate=1)
        decoder2 = self.upsampling_block(input_tensor=decoder3, concat_tensor=encoder2, num_filters=128, padding='same')
        decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)

        encoder1 = self.conv_bn_relu(x=encoder1, filters=32, kernel_size=[1, 1], dilation_rate=1)
        decoder1 = self.upsampling_block(input_tensor=decoder2, concat_tensor=encoder1, num_filters=64, padding='same')
        decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)

        outputs = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(decoder1)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(outputs)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        final_model = model
        if self.l2_regularization > 0.0:
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization)
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.bias_regularizer = tf.keras.regularizers.l2(self.l2_regularization)

            model.save("temp_model.hdf5")
            print("Temporary model with regularization is saved")
            final_model = tf.keras.models.load_model("temp_model.hdf5")
            print("Temporary model with regularization is loaded")

        self.model = final_model

    def get_model(self):

        return self.model


# class MobilenetV2SpatialPyramid:
#     def __init__(self, input_shape, optimizer):
#         self.input_shape = input_shape
#         self.optimizer = optimizer
#
#         self.encoder_spatial_dropout_rate = 0.25
#         self.decoder_spatial_dropout_rate = 0.25
#         self.l2_regularization = 0.0
#
#         self.model = None
#         self.build_model()
#
#     def build_pretrained_model(self, input_layer, use_residual):
#
#         encoder_model = tf.keras.applications.MobileNetV2(input_shape=None,
#                                                           alpha=1.4,
#                                                           include_top=False,
#                                                           weights='imagenet',
#                                                           input_tensor=input_layer)
#
#         for layer in encoder_model.layers:
#             layer.trainable = False
#
#         if use_residual:
#             encoder1 = encoder_model.get_layer('expanded_conv_add').output
#             encoder2 = encoder_model.get_layer('block_2_add').output
#             encoder3 = encoder_model.get_layer('block_5_add').output
#             encoder4 = encoder_model.get_layer('block_12_add').output
#             center = encoder_model.get_layer('out_relu').output
#         else:
#             encoder1 = encoder_model.get_layer('expanded_conv_depthwise_relu').output
#             encoder2 = encoder_model.get_layer('block_2_depthwise_relu').output
#             encoder3 = encoder_model.get_layer('block_5_depthwise_relu').output
#             encoder4 = encoder_model.get_layer('block_12_depthwise_relu').output
#             center = encoder_model.get_layer('out_relu').output
#
#         encoder1 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder1)
#         encoder2 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder2)
#         encoder3 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder3)
#         encoder4 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder4)
#         center = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(center)
#
#         return encoder1, encoder2, encoder3, encoder4, center
#
#     def conv_bn_relu(self, x, filters, kernel_size, dilation_rate):
#         x = layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.ReLU()(x)
#         return x
#
#     def spatial_pyramid_block(self, spatial_pyramid_input, filters, dilation_rates):
#
#         b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[1, 1], dilation_rate=dilation_rates[0])
#         b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[1])
#         b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rates[2])
#
#         b3 = layers.AveragePooling2D(pool_size=[4, 4], padding='valid')(spatial_pyramid_input)
#         b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
#         b3 = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(b3)
#
#         spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
#         spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=filters * 2, kernel_size=[1, 1], dilation_rate=1)
#         return spatial_pyramid_output
#
#     def decoder_block(self, input_tensor, concat_tensor, num_filters, padding):
#         decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='valid')(input_tensor)
#         decoder = layers.BatchNormalization()(decoder)
#         decoder = layers.Activation('relu')(decoder)
#
#         decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
#         decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
#         decoder = layers.BatchNormalization()(decoder)
#         decoder = layers.Activation('relu')(decoder)
#         decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
#         decoder = layers.BatchNormalization()(decoder)
#         decoder = layers.Activation('relu')(decoder)
#         return decoder
#
#     def upsampling_block(self, input_tensor, concat_tensor, num_filters, padding):
#         decoder = layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(input_tensor)
#         decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
#         decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
#         decoder = layers.BatchNormalization()(decoder)
#         decoder = layers.Activation('relu')(decoder)
#         return decoder
#
#     def build_model(self):
#         # # with deconv
#         # decoder1 = self.decoder_block(input_tensor=spatial_pyramid_output, concat_tensor=encoder4, num_filters=728, padding='same')
#         # decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)
#         # decoder2 = self.decoder_block(input_tensor=decoder1, concat_tensor=encoder3, num_filters=256, padding='same')
#         # decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)
#         # decoder3 = self.decoder_block(input_tensor=decoder2, concat_tensor=encoder2, num_filters=128, padding='valid')
#         # decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)
#         # decoder4 = self.decoder_block(input_tensor=decoder3, concat_tensor=encoder1, num_filters=64, padding='same')
#         # decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)
#
#         inputs = layers.Input(shape=self.input_shape)
#
#         encoder1, encoder2, encoder3, encoder4, center = self.build_pretrained_model(inputs, use_residual=False)
#
#         encoder4_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder4, filters=128, dilation_rates=[1, 3, 5])
#         encoder4_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder4_pyramid)
#
#         center_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=center, filters=192, dilation_rates=[1, 2, 3])
#         center_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(center_pyramid)
#
#         # without deconv
#         decoder4 = self.upsampling_block(input_tensor=center_pyramid, concat_tensor=encoder4_pyramid, num_filters=256, padding='same')
#         decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)
#
#         encoder3 = self.conv_bn_relu(x=encoder3, filters=96, kernel_size=[1, 1], dilation_rate=1)
#         decoder3 = self.upsampling_block(input_tensor=decoder4, concat_tensor=encoder3, num_filters=192, padding='same')
#         decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)
#
#         encoder2 = self.conv_bn_relu(x=encoder2, filters=64, kernel_size=[1, 1], dilation_rate=1)
#         decoder2 = self.upsampling_block(input_tensor=decoder3, concat_tensor=encoder2, num_filters=128, padding='same')
#         decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)
#
#         encoder1 = self.conv_bn_relu(x=encoder1, filters=32, kernel_size=[1, 1], dilation_rate=1)
#         decoder1 = self.upsampling_block(input_tensor=decoder2, concat_tensor=encoder1, num_filters=64, padding='same')
#         decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)
#
#         outputs = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(decoder1)
#         outputs = layers.BatchNormalization()(outputs)
#         outputs = layers.Activation('relu')(outputs)
#         outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
#         outputs = layers.BatchNormalization()(outputs)
#         outputs = layers.Activation('relu')(outputs)
#         outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
#         outputs = layers.BatchNormalization()(outputs)
#         outputs = layers.Activation('relu')(outputs)
#
#         outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(outputs)
#
#         model = models.Model(inputs=[inputs], outputs=[outputs])
#
#         final_model = model
#         if self.l2_regularization > 0.0:
#             for layer in model.layers:
#                 if hasattr(layer, 'kernel_regularizer'):
#                     layer.kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization)
#                 if hasattr(layer, 'bias_regularizer') and layer.use_bias:
#                     layer.bias_regularizer = tf.keras.regularizers.l2(self.l2_regularization)
#
#             model.save("temp_model.hdf5")
#             print("Temporary model with regularization is saved")
#             final_model = tf.keras.models.load_model("temp_model.hdf5")
#             print("Temporary model with regularization is loaded")
#
#         self.model = final_model
#
#     def get_model(self):
#         return self.model

class MobilenetV2SpatialPyramid:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.encoder_spatial_dropout_rate = 0.0
        self.decoder_spatial_dropout_rate = 0.0
        self.l2_regularization = 0.0

        self.model = None
        self.build_model()

    def build_pretrained_model(self, input_layer, use_residual):

        encoder_model = tf.keras.applications.MobileNetV2(input_shape=None,
                                                          alpha=1.4,
                                                          include_top=False,
                                                          weights='imagenet',
                                                          input_tensor=input_layer)

        for layer in encoder_model.layers:
            layer.trainable = False

        if use_residual:
            encoder1 = encoder_model.get_layer('expanded_conv_add').output
            encoder2 = encoder_model.get_layer('block_2_add').output
            encoder3 = encoder_model.get_layer('block_5_add').output
            encoder4 = encoder_model.get_layer('block_12_add').output
            center = encoder_model.get_layer('out_relu').output
        else:
            encoder1 = encoder_model.get_layer('expanded_conv_depthwise_relu').output
            encoder2 = encoder_model.get_layer('block_2_depthwise_relu').output
            encoder3 = encoder_model.get_layer('block_5_depthwise_relu').output
            encoder4 = encoder_model.get_layer('block_12_depthwise_relu').output
            center = encoder_model.get_layer('out_relu').output

        encoder1 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder1)
        encoder2 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder2)
        encoder3 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder3)
        encoder4 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder4)
        center = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(center)

        return encoder1, encoder2, encoder3, encoder4, center

    def conv_bn_relu(self, x, filters, kernel_size, dilation_rate):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def spatial_pyramid_block(self, spatial_pyramid_input):

        filters = 192
        b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=2)
        b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=3)

        b3 = layers.AveragePooling2D(pool_size=[4, 4], padding='valid')(spatial_pyramid_input)
        b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b3 = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(b3)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=384, kernel_size=[1, 1], dilation_rate=1)
        return spatial_pyramid_output

    def decoder_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='valid')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)

        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def upsampling_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        encoder1, encoder2, encoder3, encoder4, spatial_pyramid_input = self.build_pretrained_model(inputs, use_residual=False)

        spatial_pyramid_output = self.spatial_pyramid_block(spatial_pyramid_input=spatial_pyramid_input)
        spatial_pyramid_output = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(spatial_pyramid_output)

        # without deconv
        encoder4 = self.conv_bn_relu(x=encoder4, filters=128, kernel_size=[1, 1], dilation_rate=1)
        decoder4 = self.upsampling_block(input_tensor=spatial_pyramid_output, concat_tensor=encoder4, num_filters=256, padding='same')
        decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)

        encoder3 = self.conv_bn_relu(x=encoder3, filters=96, kernel_size=[1, 1], dilation_rate=1)
        decoder3 = self.upsampling_block(input_tensor=decoder4, concat_tensor=encoder3, num_filters=192, padding='same')
        decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)

        encoder2 = self.conv_bn_relu(x=encoder2, filters=64, kernel_size=[1, 1], dilation_rate=1)
        decoder2 = self.upsampling_block(input_tensor=decoder3, concat_tensor=encoder2, num_filters=128, padding='same')
        decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)

        encoder1 = self.conv_bn_relu(x=encoder1, filters=32, kernel_size=[1, 1], dilation_rate=1)
        decoder1 = self.upsampling_block(input_tensor=decoder2, concat_tensor=encoder1, num_filters=64, padding='same')
        decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)

        # # with deconv
        # decoder1 = self.decoder_block(input_tensor=spatial_pyramid_output, concat_tensor=encoder4, num_filters=728, padding='same')
        # decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)
        # decoder2 = self.decoder_block(input_tensor=decoder1, concat_tensor=encoder3, num_filters=256, padding='same')
        # decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)
        # decoder3 = self.decoder_block(input_tensor=decoder2, concat_tensor=encoder2, num_filters=128, padding='valid')
        # decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)
        # decoder4 = self.decoder_block(input_tensor=decoder3, concat_tensor=encoder1, num_filters=64, padding='same')
        # decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)

        outputs = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(decoder1)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(outputs)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        final_model = model
        if self.l2_regularization > 0.0:
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization)
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.bias_regularizer = tf.keras.regularizers.l2(self.l2_regularization)

            model.save("temp_model.hdf5")
            print("Temporary model with regularization is saved")
            final_model = tf.keras.models.load_model("temp_model.hdf5")
            print("Temporary model with regularization is loaded")

        self.model = final_model

    def get_model(self):
        return self.model

class XceptionSpatialPyramidArda:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.encoder_spatial_dropout_rate = 0.0
        self.decoder_spatial_dropout_rate = 0.0
        self.l2_regularization = 0.0

        self.model = None
        self.build_model()

    def build_pretrained_xception_model(self, input_layer):
        encoder_model = tf.keras.applications.Xception(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=input_layer,
                                                       input_shape=self.input_shape,
                                                       pooling=None,
                                                       classes=None)

        # encoder_model.load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
        for layer in encoder_model.layers:
            layer.trainable = False

        encoder1 = encoder_model.get_layer('block2_sepconv2_bn').output
        encoder2 = encoder_model.get_layer('block3_sepconv2_bn').output
        encoder3 = encoder_model.get_layer('block4_sepconv2_bn').output
        encoder4 = encoder_model.get_layer('block13_sepconv2_bn').output
        spatial_pyramid_input = encoder_model.output

        encoder1 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder1)
        encoder2 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder2)
        encoder3 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder3)
        encoder4 = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(encoder4)
        spatial_pyramid_input = layers.SpatialDropout2D(rate=self.encoder_spatial_dropout_rate)(spatial_pyramid_input)

        return encoder1, encoder2, encoder3, encoder4, spatial_pyramid_input

    def conv_bn_relu(self, x, filters, kernel_size, dilation_rate):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def spatial_pyramid_block(self, spatial_pyramid_input, filters, dilation_rates, kernel_sizes):

        b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[kernel_sizes[0], kernel_sizes[0]], dilation_rate=dilation_rates[0])
        b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[kernel_sizes[1], kernel_sizes[1]], dilation_rate=dilation_rates[1])
        b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[kernel_sizes[2], kernel_sizes[2]], dilation_rate=dilation_rates[2])

        b3 = layers.AveragePooling2D(pool_size=[4, 4], padding='valid')(spatial_pyramid_input)
        b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b3 = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(b3)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=filters * 2, kernel_size=[1, 1], dilation_rate=1)
        return spatial_pyramid_output

    def decoder_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='valid')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)

        _, concat_tensor_image_size, _, _ = concat_tensor.shape.as_list()
        _, decoder_image_size, _, _ = decoder.shape.as_list()
        if concat_tensor_image_size != decoder_image_size:
            pad_begin = int(np.ceil((decoder_image_size - concat_tensor_image_size) / 2.0))
            pad_end = int(np.floor((decoder_image_size - concat_tensor_image_size) / 2.0))
            concat_tensor = layers.ZeroPadding2D(padding=((pad_begin, pad_end), (pad_begin, pad_end)))(concat_tensor)

        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def upsampling_block(self, input_tensor, concat_tensor, num_filters, padding):
        decoder = layers.UpSampling2D(size=[2, 2], interpolation='bilinear')(input_tensor)

        _, concat_tensor_image_size, _, _ = concat_tensor.shape.as_list()
        _, decoder_image_size, _, _ = decoder.shape.as_list()
        if concat_tensor_image_size != decoder_image_size:
            pad_begin = int(np.ceil((decoder_image_size - concat_tensor_image_size) / 2.0))
            pad_end = int(np.floor((decoder_image_size - concat_tensor_image_size) / 2.0))
            concat_tensor = layers.ZeroPadding2D(padding=((pad_begin, pad_end), (pad_begin, pad_end)))(concat_tensor)

        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.Conv2D(num_filters, (3, 3), padding=padding)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        zero_padded_inputs = layers.ZeroPadding2D(padding=(3, 3))(inputs)

        encoder1, encoder2, encoder3, encoder4, center = self.build_pretrained_xception_model(zero_padded_inputs)

        input_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=inputs, filters=16, kernel_sizes=[5, 5, 5], dilation_rates=[1, 2, 3])
        input_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(input_pyramid)

        encoder1_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder1, filters=32, kernel_sizes=[1, 3, 3], dilation_rates=[1, 5, 9])
        encoder1_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder1_pyramid)

        encoder2_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder2, filters=64, kernel_sizes=[1, 3, 3], dilation_rates=[1, 5, 9])
        encoder2_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder2_pyramid)

        encoder3_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder3, filters=96, kernel_sizes=[1, 3, 3], dilation_rates=[1, 3, 5])
        encoder3_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder3_pyramid)

        encoder4_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=encoder4, filters=160, kernel_sizes=[1, 3, 3], dilation_rates=[1, 3, 5])
        encoder4_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(encoder4_pyramid)

        center_pyramid = self.spatial_pyramid_block(spatial_pyramid_input=center, filters=224, kernel_sizes=[1, 3, 3], dilation_rates=[1, 2, 3])
        center_pyramid = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(center_pyramid)

        # without deconv
        decoder4 = self.upsampling_block(input_tensor=center_pyramid, concat_tensor=encoder4_pyramid, num_filters=256, padding='same')
        decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)

        decoder3 = self.upsampling_block(input_tensor=decoder4, concat_tensor=encoder3_pyramid, num_filters=192, padding='same')
        decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)

        decoder2 = self.upsampling_block(input_tensor=decoder3, concat_tensor=encoder2_pyramid, num_filters=128, padding='same')
        decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)

        decoder1 = self.upsampling_block(input_tensor=decoder2, concat_tensor=encoder1_pyramid, num_filters=64, padding='same')
        decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)

        outputs = self.upsampling_block(input_tensor=decoder1, concat_tensor=input_pyramid, num_filters=32, padding='same')
        outputs = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(outputs)

        outputs = layers.Conv2D(32, (3, 3), padding='same')(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.Activation('relu')(outputs)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(outputs)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        final_model = model
        if self.l2_regularization > 0.0:
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization)
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.bias_regularizer = tf.keras.regularizers.l2(self.l2_regularization)

            model.save("temp_model.hdf5")
            print("Temporary model with regularization is saved")
            final_model = tf.keras.models.load_model("temp_model.hdf5")
            print("Temporary model with regularization is loaded")

        self.model = final_model

    def get_model(self):

        return self.model
