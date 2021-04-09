import warnings
import numpy as np

from keras.models import Model
from keras.layers import Input, MaxPooling2D
from keras.layers import Conv2D, Flatten, Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense


class Mynet:
    def __init__(self, size=512, num_classes=4):
        self.model = self.build_model(Input(shape=(size, size, 1)), num_classes)
        self.model.summary()

    def conv_bn(self, x, filters, kernel_size, stride, padding='valid'):
        x = Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size],
                   strides=[stride, stride], padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

        return x

    def build_model(self, inputs, num_classes):
        x = self.conv_bn(inputs, 8, 3, 1)
        x = self.conv_bn(x, 8, 3, 1)
        x = self.conv_bn(x, 16, 3, 1)
        x = self.conv_bn(x, 16, 3, 1)
        x = self.conv_bn(x, 32, 3, 1)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(units=num_classes, activation='softmax')(x)

        return Model(inputs, x)


if __name__ == '__main__':
    my_model = Mynet().model
