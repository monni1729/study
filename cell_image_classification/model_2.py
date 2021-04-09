from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import keras
import matplotlib.pyplot as plt
import skimage
import os

CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']
SIZE_MODE = 512


def build_model(load_dir):
    if load_dir:
        model = models.load_model(load_dir)
    else:
        kernel = (3, 3)
        pad = 'same'  # 'valid'
        model = models.Sequential()
        model.add(
            layers.Conv2D(8, kernel, kernel_regularizer=l2(0.001), input_shape=(SIZE_MODE, SIZE_MODE, 1), padding=pad))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(8, kernel, kernel_regularizer=l2(0.001), padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(16, kernel, kernel_regularizer=l2(0.001), padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(16, kernel, kernel_regularizer=l2(0.001), padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, kernel, kernel_regularizer=l2(0.001), padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, kernel, kernel_regularizer=l2(0.001), padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

    model.summary()
    return model


def train(model, save, graph, shear_opt=False):
    if shear_opt:
        train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,fill_mode="constant",
                                       rotation_range=10, shear_range=shear_opt, zoom_range=0.1, horizontal_flip=True,
                                       rescale=1. / 255)
    else:
        train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,fill_mode="constant",
                                       rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    if SIZE_MODE == 1024:
        bat_size = 4
    else:
        bat_size = 16

    train_generator = train_gen.flow_from_directory("C:/tf_env/dream/" + str(SIZE_MODE) + "_images/train",
                                                    classes=CLASS_NAMES, target_size=(SIZE_MODE, SIZE_MODE),
                                                    batch_size=bat_size, class_mode='categorical', color_mode='grayscale')
    val_generator = val_gen.flow_from_directory("C:/tf_env/dream/" + str(SIZE_MODE) + "_images/val",
                                                classes=CLASS_NAMES, target_size=(SIZE_MODE, SIZE_MODE), batch_size=bat_size,
                                                class_mode='categorical', color_mode='grayscale')

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.0001

    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                      callbacks.LearningRateScheduler(scheduler)]
    history = model.fit_generator(train_generator, callbacks=callbacks_list, steps_per_epoch=50, epochs=1000, verbose=2,
                                  validation_data=val_generator, validation_steps=50)

    if save:
        model.save(str(SIZE_MODE)+"_size.h5")

    if graph:
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
        # plt.show()
        plt.savefig(str(SIZE_MODE)+"_size_acc.png")
        plt.clf()

        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        plt.plot(epochs, loss, label='Training')
        plt.plot(epochs, val_loss, label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='lower right')
        # plt.show()
        plt.savefig(str(SIZE_MODE)+"_size_loss.png")
        plt.clf()


def predict(load_dir):
    model = models.load_model(load_dir)
    total_right = 0
    total_wrong = 0
    root_dir = "C:/tf_env/dream/"+str(SIZE_MODE)+"_images/val/"
    for idx, cl in enumerate(CLASS_NAMES):
        print(cl)
        os.chdir(root_dir + cl)
        files_list = os.listdir()
        print(len(files_list))
        for f in files_list:
            print(f, end='')
            data = skimage.io.imread(f)
            print(data.shape, data[0, 0])
            data = data / 255.
            data = data.reshape((1, SIZE_MODE, SIZE_MODE, 1))
            result = model.predict_classes(data)
            # result = model.predict(data)
            print(result, end='')
            if result[0] == idx:
                total_right += 1
                print("O")
            else:
                total_wrong += 1
                print("X")
        print()
    print(total_right, total_wrong, total_right / (total_wrong + total_right))
    os.chdir(root_dir)
    return total_right, total_wrong, total_right / (total_wrong + total_right)


def execute():
    # model = build_model(load_dir=None)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
    # train(model, save=True, graph=True)
    # r = predict(str(SIZE_MODE)+"_size.h5")
    r = predict("shear_10.h5")
    print(r)


if __name__ == '__main__':
    execute()
