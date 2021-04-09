from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.layers.merge import add
from keras import backend
from keras.utils import plot_model


x = Input((128, 128, 3))

r = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
r = Conv2D(256, (3, 3), strides=(1, 1), padding="same", use_bias=False)(r)
r = Conv2D(256, (1, 1), strides=(1, 1), padding="same", use_bias=False)(r)

sx = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
xr = add([sx, r])

print(backend.int_shape(sx))
print(backend.int_shape(r))
model = Model(inputs=x, outputs=xr)
model.summary()
plot_model(model, to_file="visual_1.png")
