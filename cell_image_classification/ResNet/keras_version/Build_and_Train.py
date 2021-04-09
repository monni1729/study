# Inspired by "https://github.com/raghakot/keras-resnet"


from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization as BN
from keras.regularizers import l2
from keras import backend, callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3
CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']


def _bn_relu(x):
    x = BN(axis=CHANNEL_AXIS)(x)
    return Activation("relu")(x)


def _bn_relu_conv(**params):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    strides = params.setdefault("strides", (1, 1))
    kernel_initializer = params.setdefault("kernel_initializer", "he_normal")
    padding = params.setdefault("padding", "same")
    kernel_regularizer = params.setdefault("kernel_regularizer", l2(1.e-4))

    def fn(x):
        x = _bn_relu(x)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(x)

    return fn


def _shortcut(x, residual):
    x_shape = backend.int_shape(x)
    r_shape = backend.int_shape(residual)

    stride_w = int(round(x_shape[ROW_AXIS] / r_shape[ROW_AXIS]))
    stride_h = int(round(x_shape[COL_AXIS] / r_shape[COL_AXIS]))
    equal_channels = x_shape[CHANNEL_AXIS] == r_shape[CHANNEL_AXIS]

    shortcut = x
    if stride_w > 1 or stride_h > 1 or not equal_channels:
        shortcut = Conv2D(filters=r_shape[CHANNEL_AXIS], kernel_size=(1, 1),
                          strides=(stride_w, stride_h), padding="valid",
                          kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(x)

    return add([shortcut, residual])


def _residual_block(block_type, filters, repetitions, is_first_layer=False):
    def fn(x):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)

            x = block_type(filters=filters, init_strides=init_strides,
                           is_first_block_of_first_layer=(is_first_layer and i == 0))(x)

        return x

    return fn


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def fn(x):
        if is_first_block_of_first_layer:
            conv_11 = Conv2D(filters=filters, kernel_size=(1, 1),
                             strides=init_strides, padding="same",
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(1.e-4))(x)
        else:
            conv_11 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                    strides=init_strides)(x)

        conv_33 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_11)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_33)

        return _shortcut(x, residual)

    return fn


class ResNet:
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions):
        inputs = Input(shape=input_shape)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(inputs)
        x = _bn_relu(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        filters = 64
        for i, r in enumerate(repetitions):
            x = _residual_block(block_type, filters=filters, repetitions=r, is_first_layer=(i == 0))(x)
            filters *= 2
        x = _bn_relu(x)

        # Classifier
        block_shape = backend.int_shape(x)
        x = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(x)
        x = Flatten()(x)
        x = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax")(x)
        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def build50(input_shape, num_outputs):
        return ResNet.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])


def train():
    img_rows, img_cols = 256, 256
    img_channels = 1

    lr_reducer = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = callbacks.EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = callbacks.CSVLogger('resnet50_4cells.csv')

    batch_size = 16
    nb_classes = 4
    nb_epoch = 200

    model = ResNet.build50((img_rows, img_cols, img_channels), nb_classes)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,
                                   fill_mode="constant", rotation_range=10, shear_range=10, zoom_range=0.1,
                                   horizontal_flip=True, rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory("C:/tf_env/ResNet/256_images/train/",
                                                    classes=CLASS_NAMES, target_size=(img_rows, img_cols),
                                                    batch_size=batch_size, class_mode='categorical',
                                                    color_mode='grayscale')
    val_generator = val_gen.flow_from_directory("C:/tf_env/ResNet/256_images/val/",
                                                classes=CLASS_NAMES, target_size=(img_rows, img_cols),
                                                batch_size=batch_size, class_mode='categorical', color_mode='grayscale')

    history = model.fit_generator(train_generator, callbacks=[lr_reducer, early_stopper, csv_logger],
                                  steps_per_epoch=50, epochs=nb_epoch, verbose=1,
                                  validation_data=val_generator, validation_steps=50)

    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Training')
    plt.plot(epochs, val_acc, label='Validation')
    plt.title('Model Acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig("acc.png")
    plt.clf()

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig("loss.png")
    plt.clf()


if __name__ == "__main__":
    train()

