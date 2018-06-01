import tensorflow as tf
import numpy as np
from preprocess_data import LOADED_DATASETS, load_dataset
from consts import *


def batch_iterator(dataset_name, sample_type, batch_size, is_random=True, random_seed=None):
    """
    :param dataset_name:
    :type dataset_name:
    :param sample_type:     0: test,
                            1: full train set,
                            2: train set without validation part,
                            3: test validation (intersects with test set),
                            4: train validation (subset of train set, but does not intersect with it)
                            5: train set without validation part 1
                            6: train set without validation part 2
    :type sample_type:
    :param batch_size:
    :type batch_size:
    :param is_random:
    :type is_random:
    :param random_seed:
    :type random_seed:
    :return:
    :rtype:
    """

    X, Y = LOADED_DATASETS[dataset_name][sample_type]

    N = LOADED_DATASETS[dataset_name][sample_type][0].shape[0]

    if is_random:
        if random_seed is not None:
            after_seed = np.random.randint(1 << 30)
            np.random.seed(random_seed)
            perm = np.random.permutation(N)
            np.random.seed(after_seed)
        else:
            perm = np.random.permutation(N)

        X_shuffled = X[perm]
        Y_shuffled = Y[perm]
    else:
        X_shuffled = X
        Y_shuffled = Y

    l = 0
    end = False
    while not end:
        r = min(l + batch_size, N)

        X_batch = X_shuffled[l:r]
        Y_batch = Y_shuffled[l:r]

        yield X_batch, Y_batch

        l = r
        end = r == N
