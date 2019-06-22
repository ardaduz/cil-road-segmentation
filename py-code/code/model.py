from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf


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

        self.model.compile(optimizer=optimizer,
                           loss=LossesMetrics.bce_dice_loss,
                           metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])

    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

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

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        # 256

        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)
        # 128

        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)
        # 64

        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)
        # 32

        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)
        # 16

        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)
        # 8

        center = self.conv_block(encoder4_pool, 1024)
        # center

        decoder4 = self.decoder_block(center, encoder4, 512)
        # 16

        decoder3 = self.decoder_block(decoder4, encoder3, 256)
        # 32

        decoder2 = self.decoder_block(decoder3, encoder2, 128)
        # 64

        decoder1 = self.decoder_block(decoder2, encoder1, 64)
        # 128

        decoder0 = self.decoder_block(decoder1, encoder0, 32)
        # 256

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        self.model = model

    def get_compiled_model(self):
        return self.model


class EncoderMLPModel:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape

        self.model = None

        self.build_model()

        self.model.compile(optimizer=optimizer,
                           loss=LossesMetrics.bce_dice_loss,
                           metrics=[LossesMetrics.dice_loss, LossesMetrics.root_mean_squared_error])
