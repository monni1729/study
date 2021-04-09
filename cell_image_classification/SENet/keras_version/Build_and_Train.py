from keras.layers import BatchNormalization as bn
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import Input
from keras.layers import GlobalAvgPool2D
from keras.layers import Dense
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers import Multiply
from keras.layers import Reshape
from keras import backend, callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

SE_flag = True
INIT_FILTERS = 16
DECAY = 1.e-4
CH_AXIS = 3
CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']


def _bn_relu():
    def f(x):
        x = bn(axis=CH_AXIS)(x)
        return Activation('relu')(x)

    return f


def _bn_relu_conv(filters, k, s=1):
    def f(x):
        activation = _bn_relu()(x)
        conv = Conv2D(filters, k, strides=s, padding="same", kernel_regularizer=l2(DECAY),
                      kernel_initializer="he_normal", )(activation)
        return conv

    return f


def _conv_bn_relu(filters, kernel_size, s=(1, 1)):
    def fn(x):
        x = Conv2D(filters, kernel_size, strides=s, use_bias=False,
                   padding="same", kernel_regularizer=l2(DECAY),
                   kernel_initializer="he_normal", )(x)
        x = bn(axis=CH_AXIS)(x)
        x = Activation('relu')(x)
        return x

    return fn


def _shortcut(x, residual):
    x_shape = backend.int_shape(x)
    res_shape = backend.int_shape(residual)
    size_equality = int(round(x_shape[1] / res_shape[1]))
    print("size_equality", size_equality)
    channel_equality = (x_shape[3] == res_shape[3])
    y = x
    if size_equality > 1 or not channel_equality:
        y = Conv2D(res_shape[3], (1, 1), strides=(size_equality, size_equality),
                   padding='same', kernel_regularizer=l2(DECAY),
                   kernel_initializer="he_normal", )(x)
    return Add()([y, residual])


def _resblock(block, filters, stage, first_stage=False):
    def f(x):
        for i in range(stage):
            strides = (1, 1)
            if i == 0 and not first_stage:
                strides = (2, 2)
            x = block(filters=filters, strides=strides,
                      fst_layer_fst_stage=(i == 0 and first_stage))(x)
        return x

    return f


def _bottleneck_block(filters, strides, fst_layer_fst_stage):
    def f(x):
        if fst_layer_fst_stage:
            conv1 = Conv2D(filters, (1, 1), strides=1,
                           padding='same', use_bias=False,
                           kernel_regularizer=l2(DECAY),
                           kernel_initializer="he_normal", )(x)
        else:
            conv1 = _bn_relu_conv(filters, 1, s=strides)(x)
        conv2 = _bn_relu_conv(filters, 3, s=1)(conv1)
        residual = _bn_relu_conv(filters * 4, 1, s=1)(conv2)

        se_out = residual
        if filters < INIT_FILTERS * 8 and SE_flag:
            print("filters < INIT_FILTERS * 8 and SE_flag")
            se_out = se_block(residual)
        return _shortcut(x, se_out)

    return f


def seresnet(input_shape=(256, 256, 1), num_classes=4):
    block_function = _bottleneck_block
    stage = [3, 4, 6, 3]

    filters = INIT_FILTERS
    input_tensor = Input(input_shape)
    conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3))(input_tensor)

    block = conv1
    for i, r in enumerate(stage):
        block = _resblock(block_function, filters, r, i == 0)(block)
        filters *= 2

    block = _bn_relu()(block)
    avg = GlobalAvgPool2D()(block)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer="he_normal")(avg)

    model = Model(input_tensor, output)
    return model


def _default_Fsq(input):
    fsq = GlobalAvgPool2D()(input)
    return fsq


def _default_Fex(input, ratio=16):
    shape = backend.int_shape(input)
    fc1 = Dense(shape[1] // ratio)(input)
    ac1 = Activation('relu')(fc1)
    fc2 = Dense(shape[1])(ac1)
    ac2 = Activation('sigmoid')(fc2)
    return ac2


def se_block(input, Fsq=None, Fex=None, ratio=16):
    if Fsq == None:
        Fsq = _default_Fsq
    if Fex == None:
        Fex = _default_Fex
    shape = backend.int_shape(input)
    fsq = Fsq(input)
    fex = Fex(fsq, ratio)
    fex = Reshape((1, 1, shape[3]))(fex)
    scalar = Multiply()([input, fex])
    return scalar


def train():
    img_rows, img_cols = 256, 256

    lr_reducer = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = callbacks.EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = callbacks.CSVLogger('seresnet50_4cells.csv')

    batch_size = 8
    nb_epoch = 200

    model = seresnet((img_rows, img_cols, 1), num_classes=len(CLASS_NAMES))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,
                                   fill_mode="constant", rotation_range=10, shear_range=10, zoom_range=0.1,
                                   horizontal_flip=True, rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory("C:/tf_env/classification_models/SENet/256_images/train/",
                                                    classes=CLASS_NAMES, target_size=(img_rows, img_cols),
                                                    batch_size=batch_size, class_mode='categorical',
                                                    color_mode='grayscale')
    val_generator = val_gen.flow_from_directory("C:/tf_env/classification_models/SENet/256_images/val/",
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
