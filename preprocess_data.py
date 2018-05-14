import csv
import pickle
import numpy as np

from consts import *

LOADED_DATASETS = {}


def read_csv_columns(filename, columns):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        result = []
        for row in csvreader:
            new_row = [row[0], row[1] + '. ' + row[2]]
            # new_row = [row[0], row[2]]

            result.append(new_row)

        return result


def get_token_by_char(ch):
    return ALPHABET_DICT.get(ch, UNKNOWN_TOKEN)


def preprocess_text(text):
    text = text.lower()

    # deal with leading spaces
    l = 0
    while l < len(text) and text[l] == ' ':
        l += 1

    # deal with ending spaces
    r = len(text)
    while r > 0 and text[r - 1] == ' ':
        r -= 1

    x = np.full(TEXT_SIZE, PADDING_TOKEN)

    for i in range(l, r):
        if i - l >= TEXT_SIZE:
            break

        x[i - l] = get_token_by_char(text[i])

    return x


def preprocess_dataset(dataset_name, sample_type):
    filename = original_datasets_folder + dataset_name + '/' + sample_type + '.csv'
    read_data = read_csv_columns(filename, DATASET_COLUMNS[dataset_name])
    N = len(read_data)

    Y = np.zeros((N, DATASET_NCLASSES[dataset_name]), dtype=np.int32)
    X = np.empty((N, TEXT_SIZE), dtype=np.int32)
    for i in range(N):
        Y[i][int(read_data[i][0]) - 1] = 1

        X[i, :] = preprocess_text(read_data[i][1])

    return X, Y


def save_datasets(dataset_name):
    #
    # test
    #
    X, Y = preprocess_dataset(dataset_name, 'test')
    with open(processed_datasets_folder + dataset_name + '/test.pkl', 'wb') as file:
        pickle.dump((X, Y), file)

    #
    # validation test (intersects with test set)
    #
    N = X.shape[0]

    after_seed = np.random.randint(1 << 30)
    np.random.seed(0)
    vld_test_perm = np.random.permutation(N)[: int(N * VALIDATION_TEST_DATASET_SIZE_RATIO)]
    np.random.seed(after_seed)

    with open(processed_datasets_folder + dataset_name + '/vld_test.pkl', 'wb') as file:
        pickle.dump((X[vld_test_perm], Y[vld_test_perm]), file)

    #
    # train
    #
    X, Y = preprocess_dataset(dataset_name, 'train')
    N = X.shape[0]

    with open(processed_datasets_folder + dataset_name + '/full_train.pkl', 'wb') as file:
        pickle.dump((X, Y), file)

    #
    # validation train (does not intersect with train set)
    #
    after_seed = np.random.randint(1 << 30)
    np.random.seed(0)
    val_train_perm = np.random.permutation(N)[: int(N * VALIDATION_TRAIN_DATASET_SIZE_RATIO)]
    val_train_perm_small = np.random.permutation(len(val_train_perm))[: 1000]
    np.random.seed(after_seed)

    with open(processed_datasets_folder + dataset_name + '/vld_train.pkl', 'wb') as file:
        pickle.dump((X[val_train_perm], Y[val_train_perm]), file)

    with open(processed_datasets_folder + dataset_name + '/vld_train_small.pkl', 'wb') as file:
        pickle.dump((X[val_train_perm_small], Y[val_train_perm_small]), file)

    with open(processed_datasets_folder + dataset_name + '/partial_train.pkl', 'wb') as file:
        pickle.dump((np.delete(X, val_train_perm, axis=0), np.delete(Y, val_train_perm, axis=0)), file)


def load_datasets(dataset_name):
    # test
    with open(processed_datasets_folder + dataset_name + '/test.pkl', 'rb') as file:
        X, Y = pickle.load(file)

    if dataset_name not in LOADED_DATASETS:
        LOADED_DATASETS[dataset_name] = {0: (X, Y)}
    else:
        LOADED_DATASETS[dataset_name][0] = X, Y

    # full train
    with open(processed_datasets_folder + dataset_name + '/full_train.pkl', 'rb') as file:
        X, Y = pickle.load(file)
    LOADED_DATASETS[dataset_name][1] = X, Y

    # train without validation part
    with open(processed_datasets_folder + dataset_name + '/partial_train.pkl', 'rb') as file:
        X, Y = pickle.load(file)
    LOADED_DATASETS[dataset_name][2] = X, Y

    # validation test (intersects with test set)
    with open(processed_datasets_folder + dataset_name + '/vld_test.pkl', 'rb') as file:
        X, Y = pickle.load(file)
    LOADED_DATASETS[dataset_name][3] = X, Y

    # validation train (does not intersect with train set)
    with open(processed_datasets_folder + dataset_name + '/vld_train.pkl', 'rb') as file:
        X, Y = pickle.load(file)
    LOADED_DATASETS[dataset_name][4] = X, Y


def load_dataset(dataset_name, dataset_type):
    if dataset_type == 0:
        sample_file = 'test'
    elif dataset_type == 1:
        sample_file = 'full_train'
    elif dataset_type == 2:
        sample_file = 'partial_train'
    elif dataset_type == 3:
        sample_file = 'vld_test'
    elif dataset_type == 4:
        sample_file = 'vld_train'
    else:
        raise Exception('Invalid sample type')

    with open(processed_datasets_folder + dataset_name + '/' + sample_file + '.pkl', 'rb') as file:
        X, Y = pickle.load(file)

    if dataset_name not in LOADED_DATASETS:
        LOADED_DATASETS[dataset_name] = {dataset_type: (X, Y)}
    else:
        LOADED_DATASETS[dataset_name][dataset_type] = X, Y


if __name__ == '__main__':
    save_datasets('ag_news')