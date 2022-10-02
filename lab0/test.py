import os
import struct

import numpy as np
from numpy import hsplit

root = './mnist'
labels_path = os.path.join(root, '%s-labels-idx1-ubyte' % 't10k')

images_path = os.path.join(root, '%s-images-idx3-ubyte' % 't10k')
with open(labels_path, 'rb') as lb_path:
    magic, n = struct.unpack('>II', lb_path.read(8))
    labels = np.fromfile(lb_path, dtype=np.uint8)
with open(images_path, 'rb') as img_path:
    magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
    images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 28, 28)
img1 = images.astype(np.float32) / 255.0
lab = np.zeros((labels.size, 10))
images_path = os.path.join(root, '%s-images-idx3-ubyte' % 'train')
with open(images_path, 'rb') as img_path:
    magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
    images = np.fromfile(img_path, dtype=np.uint8).reshape(60000, 28, 28)
img2 = images.astype(np.float32) / 255.0
for i, row in enumerate(lab):
    row[labels[i]] = 1
class_count = {}
lab[1] = np.array(lab[1]).astype(str)
for x in lab[1].tolist():
    tuple(x)
class_count[lab[1]] = class_count.get(lab[1], 0) + 1
# print(np.tile(img1[1], (img2.shape[0], 1)).reshape((60000, 28, 28)))
# print(img2 - np.tile(img1[i], (img2.shape[0], 1)))
print(lab[1])
print(magic)
print(num)
print(n)
print(rows)
print(cols)
print(img1[1].shape)
print(img2.shape[0], 1)
