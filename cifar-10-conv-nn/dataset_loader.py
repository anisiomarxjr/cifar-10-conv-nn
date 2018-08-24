from pkg_resources import resource_stream
import pickle
import numpy as np


def load_dataset():
    file_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    X = np.empty((0, 3, 32, 32), int)
    Y = np.empty((0, 1), int)

    for f in file_list:
        file = resource_stream("data", f)
        with file:
            dict = pickle.load(file, encoding='bytes')
            images = np.reshape(dict[b'data'], (10000, 3, 32, 32))
            X = np.append(X, images, axis=0)
            Y = np.append(Y, np.transpose([dict[b'labels']]), axis=0)

    return X, Y
