import matplotlib.pyplot as plt
import skimage
import keras
import numpy as np
import math


def generator_sample():
    # from keras.preprocessing import image
    # datagen = image.ImageDataGenerator(vertical_flip=True, shear_range=0.2, fill_mode="constant", rescale=1. / 255)
    # fname = "C:/tf_env/dream/512_images/val/NIH/80.png"
    #
    # img = image.load_img(fname, target_size=(512, 512, 1))
    #
    # x = image.img_to_array(img)
    # x = x.reshape((1,) + x.shape)
    #
    # i = 0
    # for batch in datagen.flow(x, batch_size=1):
    #     plt.figure(i)
    #     imgplot = plt.imshow(image.array_to_img(batch[0]))
    #     i += 1
    #     break
    fname = "C:/tf_env/dream/512_images/val/NIH/80.png"

    img = image.load_img(fname, target_size=(512, 512, 1))

    x = image.img_to_array(img)
    x = keras.preprocessing.image.apply_affine_transform(x, shear=10)
    plt.imshow(x)
    plt.show()


def shear_sample():
    fname = "C:/tf_env/dream/512_images/train/NIH/1.png"
    img = skimage.io.imread(fname)
    print(img[0, 0])
    img_bg = np.zeros((512, 1024))
    for i in range(512):
        img_bg[i, i:i + 512] = img[i, :]

    # img_bg[:, :512] = img
    # afine_tf = skimage.transform.AffineTransform(shear=1.0472 * 1.5)
    # img = skimage.transform.warp(img, inverse_map=afine_tf)
    # img = img[:, 443:443 + 512]
    skimage.io.imsave("C:/tf_env/dream/temp_60.png", img_bg[:, 256: 256+512])


if __name__ == '__main__':
    shear_sample()
