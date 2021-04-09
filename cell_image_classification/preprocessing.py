import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
import skimage
import matplotlib.pyplot as plt


def preprocessing_1():
    os.chdir("C:/Users/bntlserver/Desktop/ws_01_26/NIH")
    files_list = os.listdir()
    cnt = 1
    for f in files_list:
        image = imread(f, as_gray=True)
        print("image_shape : ", image.shape)
        print("int_or_flate : ", image[0, 0])

        image_list = []
        image_list.append(image[7:7 + 512, 176:176 + 512])
        image_list.append(image[7:7 + 512, 176 + 512:176 + 1024])
        image_list.append(image[7 + 512:7 + 1024, 176:176 + 512])
        image_list.append(image[7 + 512:7 + 1024, 176 + 512:176 + 1024])

        for e in image_list:
            imsave("C:/tf_env/dream/image_temp/" + str(cnt) + ".png", e)
            cnt += 1

        if cnt == 185:
            break

    print(files_list)


def preprocessing_2(temp_dir, class_names, additional):
    # class_names = ["HELA", "MCF7", "NIH", "SK"]
    for i in range(len(class_names)):
        os.chdir("C:/tf_env/dream/val/" + class_names[i])
        files_list_original = os.listdir()

        os.chdir("C:/Users/bntlserver/Desktop/" + temp_dir + class_names[i])
        files_list_new = os.listdir()
        cnt = 1
        for f in files_list_new:
            if additional + str(cnt) + ".png" in files_list_original:

                image = imread(f, as_gray=True)
                image = image[7:7 + 1024, 176:176 + 1024]
                image = np.reshape(image, (1024, 1024, 1))

                imsave("C:/tf_env/dream/1024_images/val/" + class_names[i] + "/" +
                       additional + str(cnt) + ".png", image)
            else:
                pass
            cnt += 1


if __name__ == "__main__":
    preprocessing_2("01.31/", ["HELA", "MCF7", "NIH", "SK"], "")
    preprocessing_2("02.02/", ["HELA", "MCF7", "NIH"], "a")
    preprocessing_2("02.04/", ["HELA", "MCF7", "NIH", "SK"], "b")
    preprocessing_2("02.06/", ["HELA", "MCF7", "NIH", "SK"], "c")
