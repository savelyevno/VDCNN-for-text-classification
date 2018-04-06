# -*- coding: utf-8 -*-

TEXT_SIZE = 1024
EMBEDDING_SIZE = 16

EMBEDDING_GEN_SEED = 0

# s = "abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’\"/|_#$%ˆ&*~‘+=<>()[]{}"
ALPHABET_DICT = {
    ' ': 36,
    '!': 41,
    '"': 45,
    '#': 49,
    '$': 50,
    '%': 51,
    '&': 53,
    '(': 61,
    ')': 62,
    '*': 54,
    '+': 57,
    ',': 38,
    '-': 37,
    '.': 40,
    '/': 46,
    '0': 26,
    '1': 27,
    '2': 28,
    '3': 29,
    '4': 30,
    '5': 31,
    '6': 32,
    '7': 33,
    '8': 34,
    '9': 35,
    ':': 43,
    ';': 39,
    '<': 59,
    '=': 58,
    '>': 60,
    '?': 42,
    '[': 63,
    ']': 64,
    '_': 48,
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    '{': 65,
    '|': 47,
    '}': 66,
    '~': 55,
    'ˆ': 52,
    '‘': 56,
    '’': 44
}
UNKNOWN_TOKEN = len(ALPHABET_DICT)
PADDING_TOKEN = len(ALPHABET_DICT) + 1

CURRENT_DATASET = 'ag_news'
VALIDATION_TEST_DATASET_SIZE_RATIO = 0.1
VALIDATION_TRAIN_DATASET_SIZE_RATIO = 1/6

original_datasets_folder = 'datasets/original/'
processed_datasets_folder = 'datasets/preprocessed/'

DATASETS = [
    'dbpedia',
    'ag_news',
]

DATASET_NCLASSES = {
    'dbpedia': 14,
    'ag_news': 4,
}

DATASET_COLUMNS = {
    'dbpedia': [0, 2],
    'ag_news': [0, 2],
}
DATASET_SIZE = {
    'ag_news': 120000
}

MAXPOOL_K = 8

N_FC_HIDDEN_LAYERS = 2048
