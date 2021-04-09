import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4)
classes = ['HELA', 'MCF7', 'NIH', 'SK']
mat = np.array([[60, 0, 0, 0], [1, 59, 0, 0], [0, 1, 61, 1], [0, 0, 0, 68]], dtype=np.int)


def my_confusion_matrix():
    title = "Confusion matrix"
    cmap = plt.cm.Greens

    con_mat = np.copy(mat)
    '''
    for i in range(4):
        summ = np.sum(con_mat[i,:])
        for j in range(4):
            con_mat[i,j] /= summ
    '''

    plt.figure(figsize=(14, 12))
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    # tick_marks = np.arange(len(classes)+1)
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, con_mat[i, j], fontsize=20,
                 horizontalalignment="center",
                 color="white" if con_mat[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.show()


def bar_graph():
    # objects = ('0°', '10°', '20°', '40°', '60°')
    # performance = [0.976, 0.984, 0.976, 0.968, 0.920]
    objects = (128, 256, 512, 1024)
    performance = [0.781, 0.940, 0.984, 0.980]
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, performance, align='center', alpha=1, width=0.5, color='black')
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.ylim((0.5, 1.04))
    # plt.xlabel('Size of images')
    plt.xlabel('Resolution of images')
    # plt.title('Programming language usage')

    plt.show()


def performance_table():
    print(' acc ', ' pre ', ' sen ', ' spec ', 'f1 ')
    t_all = mat[0, 0] + mat[1, 1] + mat[2, 2] + mat[3, 3]
    for cnt in range(4):
        tp = mat[cnt, cnt]
        tn = t_all - tp
        fp = mat[cnt, 0] + mat[cnt, 1] + mat[cnt, 2] + mat[cnt, 3] - tp
        fn = mat[0, cnt] + mat[1, cnt] + mat[2, cnt] + mat[3, cnt] - tp

        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = tp / (tp + fp)
        sen = tp / (tp + fn)
        spec = tn / (fp + tn)
        f1 = 2 * tp / (2 * tp + fp + fn)

        temp = np.array([acc, pre, sen, spec, f1])
        print(temp)


if __name__ == "__main__":
    # my_confusion_matrix()
    bar_graph()
    # performance_table()
