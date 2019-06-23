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


class BaselineModel:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape

        self.model = None
        self.build_model()

    def conv_bn_relu(self, input_tensor, num_filters, kernel_size=[3, 3], dilation_rate=1):
        encoder = layers.Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same')(input_tensor)
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

    def spatial_pyramid_block(self, spatial_pyramid_input, num_filters):
        # identity
        b0 = self.conv_bn_relu(input_tensor=spatial_pyramid_input, num_filters=num_filters, kernel_size=[1, 1], dilation_rate=1)

        # feature extraction
        b1 = self.conv_bn_relu(input_tensor=spatial_pyramid_input, num_filters=num_filters, kernel_size=[3, 3], dilation_rate=1)
        b2 = self.conv_bn_relu(input_tensor=spatial_pyramid_input, num_filters=num_filters, kernel_size=[3, 3], dilation_rate=3)
        b3 = self.conv_bn_relu(input_tensor=spatial_pyramid_input, num_filters=num_filters, kernel_size=[3, 3], dilation_rate=5)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(input_tensor=spatial_pyramid_output,
                                                   num_filters=num_filters * 2,
                                                   kernel_size=[1, 1],
                                                   dilation_rate=1)
        return spatial_pyramid_output

    def build_model(self):
        dropout_rate = 0.5
        seed = 1234

        inputs = layers.Input(shape=self.input_shape)
        # 256
        # 304
        # 384

        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)
        # 128
        # 152
        # 192

        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)
        # 64
        # 76
        # 96

        encoder1_pool = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder1_pool)
        encoder1 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder1)

        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)
        # 32
        # 38
        # 48

        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)
        # 16
        # 19
        # 24

        encoder3_pool = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder3_pool)
        encoder3 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(encoder3)

        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)
        # 8
        # bad
        # 12

        # spatial pyramid
        spatial_pyramid_output = self.spatial_pyramid_block(spatial_pyramid_input=encoder4_pool, num_filters=512)

        decoder4 = self.decoder_block(spatial_pyramid_output, encoder4, 512)
        # 16
        # bad
        # 24

        decoder4 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(decoder4)

        decoder3 = self.decoder_block(decoder4, encoder3, 256)
        # 32
        # 38

        decoder2 = self.decoder_block(decoder3, encoder2, 128)
        # 64
        # 76

        decoder2 = layers.SpatialDropout2D(rate=dropout_rate, seed=seed)(decoder2)

        decoder1 = self.decoder_block(decoder2, encoder1, 64)
        # 128
        # 152

        decoder0 = self.decoder_block(decoder1, encoder0, 32)
        # 256
        # 304

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


class XceptionUNet:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.encoder_spatial_dropout_rate = 0.3
        self.decoder_spatial_dropout_rate = 0.3
        self.l2_regularization = 0.005

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

        encoder1 = encoder_model.get_layer('block1_conv2_act').output
        encoder2 = encoder_model.get_layer('block2_pool').output
        encoder3 = encoder_model.get_layer('block3_pool').output
        encoder4 = encoder_model.get_layer('block4_pool').output
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

    def spatial_pyramid_block(self, spatial_pyramid_input):

        filters = 128
        b0 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b1 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=3)
        b2 = self.conv_bn_relu(x=spatial_pyramid_input, filters=filters, kernel_size=[3, 3], dilation_rate=5)

        b3 = layers.AveragePooling2D(pool_size=[3, 3], padding='valid')(spatial_pyramid_input)
        b3 = self.conv_bn_relu(x=b3, filters=filters, kernel_size=[1, 1], dilation_rate=1)
        b3 = layers.UpSampling2D(size=[3, 3], interpolation='bilinear')(b3)

        spatial_pyramid_output = layers.concatenate([b0, b1, b2, b3])
        spatial_pyramid_output = self.conv_bn_relu(x=spatial_pyramid_output, filters=filters*2, kernel_size=[1, 1], dilation_rate=1)
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
        decoder = layers.UpSampling2D(size=[4, 4], interpolation='bilinear')(input_tensor)

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

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        encoder1, encoder2, encoder3, encoder4, spatial_pyramid_input = self.build_pretrained_xception_model(inputs)

        spatial_pyramid_output = self.spatial_pyramid_block(spatial_pyramid_input=spatial_pyramid_input)
        spatial_pyramid_output = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(spatial_pyramid_output)

        # without deconv
        encoder3 = self.conv_bn_relu(x=encoder3, filters=128, kernel_size=[1, 1], dilation_rate=1)
        decoder1 = self.upsampling_block(input_tensor=spatial_pyramid_output, concat_tensor=encoder3, num_filters=128, padding='same')
        decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)

        encoder1 = self.conv_bn_relu(x=encoder1, filters=64, kernel_size=[1, 1], dilation_rate=1)
        decoder2 = self.upsampling_block(input_tensor=decoder1, concat_tensor=encoder1, num_filters=64, padding='same')
        decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)

        # # with deconv
        # decoder1 = self.decoder_block(input_tensor=spatial_pyramid_output, concat_tensor=encoder4, num_filters=728, padding='same')
        # decoder1 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder1)
        # decoder2 = self.decoder_block(input_tensor=decoder1, concat_tensor=encoder3, num_filters=256, padding='same')
        # decoder2 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder2)
        # decoder3 = self.decoder_block(input_tensor=decoder2, concat_tensor=encoder2, num_filters=128, padding='valid')
        # decoder3 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder3)
        # decoder4 = self.decoder_block(input_tensor=decoder3, concat_tensor=encoder1, num_filters=64, padding='same')
        # decoder4 = layers.SpatialDropout2D(rate=self.decoder_spatial_dropout_rate)(decoder4)

        outputs = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid')(decoder2)
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

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                layer.add_loss(tf.keras.regularizers.l2(self.l2_regularization)(layer.kernel))
            elif isinstance(layer, tf.keras.layers.SeparableConv2D):
                layer.add_loss(tf.keras.regularizers.l2(self.l2_regularization)(layer.pointwise_kernel))
                layer.add_loss(tf.keras.regularizers.l2(self.l2_regularization)(layer.depthwise_kernel))

            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(tf.keras.regularizers.l2(self.l2_regularization)(layer.bias))

        self.model = model

    def get_model(self):

        return self.model
