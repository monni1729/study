import numpy as np
import skimage
import os

root_dir = "C:/tf_env/dream/"
sub_dir = ["train/", "val/"]
CLASS_NAMES = ['HELA/', 'MCF7/', 'NIH/', 'SK/']


def resss():
    cnt = 1
    for s in sub_dir:
        cur_dir = root_dir + s

        for c in CLASS_NAMES:
            os.chdir(cur_dir + c)
            files_list = os.listdir()

            for f in files_list:
                data = skimage.io.imread(f)
                data = np.reshape(data, (128, 128, 1))
                data = skimage.transform.rescale(data, 0.25)
                skimage.io.imsave("C:/tf_env/dream/128_images/" + s + c + str(cnt) + ".png", data)
                print("C:/tf_env/dream/128_images/" + s + c + str(cnt) + ".png")
                cnt += 1


if __name__ == "__main__":
    resss()
