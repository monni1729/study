from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend, callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

DECAY = 5e-4  # l2_decay
CH_AXIS = 3  # channel axis
CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']


def conv_bn_relu(filters, kernel, strides=(1, 1)):
    def fn(x):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, use_bias=False, padding="same",
                   kernel_initializer="he_normal", kernel_regularizer=l2(DECAY))(x)
        x = BatchNormalization(axis=CH_AXIS)(x)
        x = Activation("relu")(x)
        return x

    return fn


def inception_stem(x):
    x = conv_bn_relu(32, (3, 3), strides=(2, 2))(x)
    x = conv_bn_relu(32, (3, 3))(x)
    x = conv_bn_relu(64, (3, 3))(x)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x2 = conv_bn_relu(96, (3, 3), strides=(2, 2))(x)

    x = concatenate([x1, x2], axis=CH_AXIS)

    x1 = conv_bn_relu(64, (1, 1))(x)
    x1 = conv_bn_relu(96, (3, 3))(x1)

    x2 = conv_bn_relu(64, (1, 1))(x)
    x2 = conv_bn_relu(64, (1, 7))(x2)
    x2 = conv_bn_relu(64, (7, 1))(x2)
    x2 = conv_bn_relu(96, (3, 3))(x2)

    x = concatenate([x1, x2], axis=CH_AXIS)
    x1 = conv_bn_relu(192, (3, 3), strides=(2, 2))(x)
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = concatenate([x1, x2], axis=CH_AXIS)
    return x


def inception_a(x):
    x1 = conv_bn_relu(96, (1, 1))(x)

    x2 = conv_bn_relu(64, (1, 1))(x)
    x2 = conv_bn_relu(96, (3, 3))(x2)

    x3 = conv_bn_relu(64, (1, 1))(x)
    x3 = conv_bn_relu(96, (3, 3))(x3)
    x3 = conv_bn_relu(96, (3, 3))(x3)

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x4 = conv_bn_relu(96, (1, 1))(x4)

    x = concatenate([x1, x2, x3, x4], axis=CH_AXIS)
    return x


def inception_b(x):
    x1 = conv_bn_relu(384, (1, 1))(x)

    x2 = conv_bn_relu(192, (1, 1))(x)
    x2 = conv_bn_relu(224, (1, 7))(x2)
    x2 = conv_bn_relu(256, (7, 1))(x2)

    x3 = conv_bn_relu(192, (1, 1))(x)
    x3 = conv_bn_relu(192, (7, 1))(x3)
    x3 = conv_bn_relu(224, (1, 7))(x3)
    x3 = conv_bn_relu(224, (7, 1))(x3)
    x3 = conv_bn_relu(256, (1, 7))(x3)

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x4 = conv_bn_relu(128, (1, 1))(x4)

    x = concatenate([x1, x2, x3, x4], axis=CH_AXIS)
    return x


def inception_c(x):
    x1 = conv_bn_relu(256, (1, 1))(x)

    x2 = conv_bn_relu(384, (1, 1))(x)
    x2_1 = conv_bn_relu(256, (1, 3))(x2)
    x2_2 = conv_bn_relu(256, (3, 1))(x2)
    x2 = concatenate([x2_1, x2_2], axis=CH_AXIS)

    x3 = conv_bn_relu(384, (1, 1))(x)
    x3 = conv_bn_relu(448, (3, 1))(x3)
    x3 = conv_bn_relu(512, (1, 3))(x3)
    x3_1 = conv_bn_relu(256, (1, 3))(x3)
    x3_2 = conv_bn_relu(256, (3, 1))(x3)
    x3 = concatenate([x3_1, x3_2], axis=CH_AXIS)

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x4 = conv_bn_relu(256, (1, 1))(x4)

    x = concatenate([x1, x2, x3, x4], axis=CH_AXIS)
    return x


def reduction_a(x):
    x1 = conv_bn_relu(384, (3, 3), strides=(2, 2))(x)

    x2 = conv_bn_relu(192, (1, 1))(x)
    x2 = conv_bn_relu(224, (3, 3))(x2)
    x2 = conv_bn_relu(256, (3, 3), strides=(2, 2))(x2)

    x3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = concatenate([x1, x2, x3], axis=CH_AXIS)
    return x


def reduction_b(x):
    x1 = conv_bn_relu(192, (1, 1))(x)
    x1 = conv_bn_relu(192, (3, 3), strides=(2, 2))(x1)

    x2 = conv_bn_relu(256, (1, 1))(x)
    x2 = conv_bn_relu(256, (1, 7))(x2)
    x2 = conv_bn_relu(320, (7, 1))(x2)
    x2 = conv_bn_relu(320, (3, 3), strides=(2, 2))(x2)

    x3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x1, x2, x3], axis=CH_AXIS)
    return x


def create_inception_v4(input_shape, classes):
    init = Input(input_shape)
    x = inception_stem(init)

    for _ in range(4):
        x = inception_a(x)
    x = reduction_a(x)

    for _ in range(7):
        x = inception_b(x)
    x = reduction_b(x)

    for _ in range(3):
        x = inception_c(x)
    x = AveragePooling2D((8, 8))(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)

    out = Dense(units=classes, activation='softmax')(x)
    model = Model(init, out)
    model.summary()

    return model


def train():
    img_rows, img_cols = 256, 256

    lr_reducer = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = callbacks.EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = callbacks.CSVLogger('inception_v4_4cells.csv')

    batch_size = 16
    nb_epoch = 200

    model = create_inception_v4((img_rows, img_cols, 1), classes=len(CLASS_NAMES))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],)
    train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,
                                   fill_mode="constant", rotation_range=10, shear_range=10, zoom_range=0.1,
                                   horizontal_flip=True, rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory("C:/tf_env/classification_models/256_images/train/",
                                                    classes=CLASS_NAMES, target_size=(img_rows, img_cols),
                                                    batch_size=batch_size, class_mode='categorical',
                                                    color_mode='grayscale')
    val_generator = val_gen.flow_from_directory("C:/tf_env/classification_models/256_images/val/",
                                                classes=CLASS_NAMES, target_size=(img_rows, img_cols),
                                                batch_size=batch_size, class_mode='categorical', color_mode='grayscale')

    history = model.fit_generator(train_generator, callbacks=[lr_reducer, early_stopper, csv_logger],
                                  steps_per_epoch=30, epochs=nb_epoch, verbose=1,
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
