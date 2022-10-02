import numpy as np
from tqdm import tqdm


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # raise NotImplementedError
        num_test = X.shape[0]
        label_list = []
        for i in range(num_test):
            distances = np.sqrt(np.sum(((self.X - (np.tile(X[i],
                                                           (self.X.shape[0], 1))).reshape((60000, 28, 28)))) ** 2,
                                       axis=1))
            distances = np.mean(distances, axis=1)
            nearest_k = np.argsort(distances)
            topK = nearest_k[:self.k]
            class_count = {}
            for i in topK:
                self.y[i] = np.array(self.y[i])
                class_count[self.y[i]] = class_count.get(self.y[i], 0) + 1
            sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
            label_list.append(sorted_class_count[0][0])
        return np.array(label_list)

        # End of todo
