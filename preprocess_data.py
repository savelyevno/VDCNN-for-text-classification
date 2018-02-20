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
            new_row = []

            for i in columns:
                new_row.append(row[i])

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
    filename = original_datasets_folder + dataset_name
    if sample_type == 0:
        filename += '/test.csv'
    elif sample_type == 1:
        filename += '/train.csv'
    else:
        raise Exception('Invalid sample type')

    read_data = read_csv_columns(filename, DATASET_COLUMNS[dataset_name])

    N = len(read_data)

    Y = np.zeros((N, DATASET_NCLASSES[dataset_name]), dtype=np.int32)
    X = np.empty((N, TEXT_SIZE), dtype=np.int32)
    for i in range(N):
        Y[i][int(read_data[i][0]) - 1] = 1

        X[i, :] = preprocess_text(read_data[i][1])

    return X, Y


def save_dataset(dataset_name, sample_type):
    if sample_type == 0:
        sample_file = 'test'
    elif sample_type == 1:
        sample_file = 'train'
    else:
        raise Exception('Invalid sample type')

    X, Y = preprocess_dataset(dataset_name, sample_type)

    with open(processed_datasets_folder + dataset_name + '/' + sample_file + '.pkl', 'wb') as file:
        pickle.dump((X, Y), file)


def load_datasets(dataset_name):
    # test
    with open(processed_datasets_folder + dataset_name + '/test.pkl', 'rb') as file:
        X, Y = pickle.load(file)

    if dataset_name not in LOADED_DATASETS:
        LOADED_DATASETS[dataset_name] = {0: (X, Y)}
    else:
        LOADED_DATASETS[dataset_name][0] = X, Y

    # validation
    N = X.shape[0]

    after_seed = np.random.randint(1 << 30)
    np.random.seed(0)
    val_perm = np.random.permutation(N)[: int(N * VALIDATION_DATASET_SIZE_RATIO)]
    np.random.seed(after_seed)

    X_val = X[val_perm]
    Y_val = Y[val_perm]
    LOADED_DATASETS[dataset_name][2] = X_val, Y_val

    # train
    with open(processed_datasets_folder + dataset_name + '/train.pkl', 'rb') as file:
        X, Y = pickle.load(file)

    LOADED_DATASETS[dataset_name][1] = X, Y


def load_dataset(dataset_name, dataset_type):
    if dataset_type == 0:
        sample_file = 'test'
    elif dataset_type == 1:
        sample_file = 'train'
    else:
        raise Exception('Invalid sample type')

    with open(processed_datasets_folder + dataset_name + '/' + sample_file + '.pkl', 'rb') as file:
        X, Y = pickle.load(file)

    if dataset_name not in LOADED_DATASETS:
        LOADED_DATASETS[dataset_name] = {dataset_type: (X, Y)}
    else:
        LOADED_DATASETS[dataset_name][dataset_type] = X, Y


if __name__ == '__main__':
    save_dataset('ag_news', 0)
    save_dataset('ag_news', 1)
