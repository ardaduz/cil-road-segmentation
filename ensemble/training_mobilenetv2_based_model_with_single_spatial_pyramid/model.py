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

class MobilenetV2SpatialPyramid:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.encoder_spatial_dropout_rate = 0.25
        self.decoder_spatial_dropout_rate = 0.25
        self.l2_regularization = 0.0

        self.model = None
        self.build_model()

    def build_pretrained_model(self, input_layer, use_residual):

        encoder_model = tf.keras.applications.MobileNetV2(input_shape=None,
                                                          alpha=1.4,
                                                          include_top=False,
                                                          weights=None,
                                                          input_tensor=input_layer)

        encoder_model.load_weights("../mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5")

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
