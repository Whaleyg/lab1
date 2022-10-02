import os
import struct
import urllib

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from knn import Knn


def load_mnist(root='./mnist', kind='train'):
    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    labels_path = os.path.join(root, '%s-labels-idx1-ubyte' % kind)

    images_path = os.path.join(root, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lb_path:
        magic, n = struct.unpack('>II', lb_path.read(8))
        labels = np.fromfile(lb_path, dtype=np.uint8)
    with open(images_path, 'rb') as img_path:
        magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
        images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 28, 28)
    img = images.astype(np.float32) / 255.0
    lab = np.zeros((labels.size, 10))
    for i, row in enumerate(lab):
        row[labels[i]] = 1
    return img, labels


# End of todo


def main():
    X_train, y_train = load_mnist()
    X_test, y_test = load_mnist(root='./mnist', kind='t10k')
    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
