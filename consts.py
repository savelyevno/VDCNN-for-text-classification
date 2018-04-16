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
INVERSE_ALPHABET_DICT = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
    9: 'j',
    10: 'k',
    11: 'l',
    12: 'm',
    13: 'n',
    14: 'o',
    15: 'p',
    16: 'q',
    17: 'r',
    18: 's',
    19: 't',
    20: 'u',
    21: 'v',
    22: 'w',
    23: 'x',
    24: 'y',
    25: 'z',
    26: '0',
    27: '1',
    28: '2',
    29: '3',
    30: '4',
    31: '5',
    32: '6',
    33: '7',
    34: '8',
    35: '9',
    36: ' ',
    37: '-',
    38: ',',
    39: ';',
    40: '.',
    41: '!',
    42: '?',
    43: ':',
    44: '’',
    45: '"',
    46: '/',
    47: '|',
    48: '_',
    49: '#',
    50: '$',
    51: '%',
    52: 'ˆ',
    53: '&',
    54: '*',
    55: '~',
    56: '‘',
    57: '+',
    58: '=',
    59: '<',
    60: '>',
    61: '(',
    62: ')',
    63: '[',
    64: ']',
    65: '{',
    66: '}',
    67: '(UNK)',
    68: '(PAD)'
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
    'ag_news': 120000 * (1 - VALIDATION_TRAIN_DATASET_SIZE_RATIO)
}

DATASET_CLASS_NAMES = {
    'ag_news': {
        1: 'World',
        2: 'Sports',
        3: 'Business',
        4: 'Sci/Tech'
    }
}

MAXPOOL_K = 8

N_FC_HIDDEN_LAYERS = 2048
