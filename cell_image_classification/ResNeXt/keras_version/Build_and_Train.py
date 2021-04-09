from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend, callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

CARD = 4  # cardinality
DECAY = 5e-4  # l2_decay
CH_AXIS = 3  # channel axis
CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']


def conv_bn(filters, kernel, strides=(1, 1)):
    def fn(x):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, use_bias=False, padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(DECAY))(x)
        x = BatchNormalization(axis=CH_AXIS)(x)
        return x

    return fn


def resnext(input_shape, nb_classes, depth, width):
    img_input = Input(shape=input_shape)
    x = create_resnext(img_input, nb_classes, depth, width)

    model = Model(inputs=img_input, outputs=x, name='resnext')
    return model


def initial_conv_block_imagenet(x):
    x = conv_bn(filters=64, kernel=(7, 7), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    return x


def grouped_convolution_block(x, grouped_channels, strides):
    group_list = []
    for c in range(CARD):
        xg = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(x)
        xg = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                    kernel_initializer='he_normal', kernel_regularizer=l2(DECAY))(xg)

        group_list.append(xg)

    group_merge = concatenate(group_list, axis=CH_AXIS)
    x = BatchNormalization(axis=CH_AXIS)(group_merge)
    x = Activation('relu')(x)

    return x


def bottleneck_block(x, filters, strides=(1, 1)):
    xi = x
    if backend.int_shape(xi)[CH_AXIS] != 2 * filters:
        xi = conv_bn(filters=filters * 2, kernel=(1, 1), strides=strides)(xi)

    grouped_channels = filters // CARD
    xr = conv_bn(filters=filters, kernel=(1, 1))(x)
    xr = Activation('relu')(xr)
    xr = grouped_convolution_block(xr, grouped_channels, strides)
    xr = conv_bn(filters=filters * 2, kernel=(1, 1))(xr)

    xa = add([xi, xr])
    xa = Activation('relu')(xa)

    return xa


def create_resnext(img_input, nb_classes, depth, width):
    filters = CARD * width
    filters_list = [filters * (2 ** i) for i in range(len(depth))]

    x = initial_conv_block_imagenet(img_input)
    for idx, nb_blocks in enumerate(depth):
        for sub_idx in range(nb_blocks):
            strides = (2, 2) if idx > 0 and sub_idx == 0 else (1, 1)
            x = bottleneck_block(x, filters_list[idx], strides=strides)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(DECAY),
              kernel_initializer='he_normal', activation='softmax')(x)

    return x


def train():
    img_rows, img_cols = 256, 256

    lr_reducer = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = callbacks.EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = callbacks.CSVLogger('resnext50_4cells.csv')

    batch_size = 16
    nb_epoch = 200

    model = resnext((img_rows, img_cols, 1), nb_classes=len(CLASS_NAMES), depth=[3, 4, 6, 3], width=16)
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
    # model = resnext((256, 256, 1), nb_classes=4, depth=[3, 4, 6, 3], width=16)
    # model.summary()
