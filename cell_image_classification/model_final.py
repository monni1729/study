from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import matplotlib.pyplot as plt
import skimage
import os

CLASS_NAMES = ['HELA', 'MCF7', 'NIH', 'SK']


def build_model(image_size):
    kernel = (3, 3)
    input_shape = (image_size, image_size, 1)
    pad = 'same'  # 'valid'
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel, kernel_regularizer=l2(0.001), input_shape=input_shape, padding=pad))
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


def train(image_size, shear_opt):
    model = build_model(image_size)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])

    train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True,
                                   fill_mode="constant", rotation_range=10, shear_range=shear_opt, zoom_range=0.1,
                                   horizontal_flip=True, rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    if image_size >= 1024:
        bat_size = 4
    else:
        bat_size = 16

    train_generator = train_gen.flow_from_directory("C:/tf_env/dream/" + str(image_size) + "_images/train",
                                                    classes=CLASS_NAMES, target_size=(image_size, image_size),
                                                    batch_size=bat_size, class_mode='categorical',
                                                    color_mode='grayscale')
    val_generator = val_gen.flow_from_directory("C:/tf_env/dream/" + str(image_size) + "_images/val",
                                                classes=CLASS_NAMES, target_size=(image_size, image_size),
                                                batch_size=bat_size, class_mode='categorical', color_mode='grayscale')

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 30:  # 100
            return 0.0001
        else:
            return 0.00001

    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True), # 30
                      callbacks.LearningRateScheduler(scheduler)]

    history = model.fit_generator(train_generator, callbacks=callbacks_list, steps_per_epoch=50, epochs=1000, verbose=2,
                                  validation_data=val_generator, validation_steps=50)

    model.save(str(image_size) + "size_" + str(shear_opt) + "degree.h5")

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
    plt.savefig(str(image_size) + "size_" + str(shear_opt) + "degree_acc.png")
    plt.clf()

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(str(image_size) + "size_" + str(shear_opt) + "degree_loss.png")
    plt.clf()


def predict(load_dir, image_size):
    os.chdir("C:/tf_env/dream/")
    model = models.load_model(load_dir)
    total_right = 0
    total_wrong = 0
    root_dir = "C:/tf_env/dream/" + str(image_size) + "_images/val/"
    for idx, cl in enumerate(CLASS_NAMES):
        print(cl)
        os.chdir(root_dir + cl)
        files_list = os.listdir()
        print(len(files_list))

        for f in files_list:
            data = skimage.io.imread(f)
            print("file_name : ", f, "  shape : ", data.shape, "  which should be integer : ", data[0, 0])
            data = data / 255.
            data = data.reshape((1, image_size, image_size, 1))
            result = model.predict_classes(data)
            print(result, end='')
            if result[0] == idx:
                total_right += 1
                print(" O")
            else:
                total_wrong += 1
                print(" X")
        print()
    print(total_right, total_wrong, total_right / (total_wrong + total_right))

    os.chdir("C:/tf_env/dream/")
    return total_right, total_wrong, total_right / (total_wrong + total_right)


def execute():
    image_size = 512
    shear_opt = 10

    # train(image_size, shear_opt)
    r = predict(str(image_size) + "size_" + str(shear_opt) + "degree.h5", image_size)
    print(r)


if __name__ == '__main__':
    execute()
