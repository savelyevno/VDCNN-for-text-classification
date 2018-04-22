import os
import argparse
import pickle

from train import train
from test import test
from test_params import test_params
from NetworkParams import NetworkParams


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_parser_args():
    parser = argparse.ArgumentParser('VeryDeepConvolutionalNeuralNetwork for text classification',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(to_continue=False)
    parser.set_defaults(to_train=False)
    parser.set_defaults(to_test=False)
    parser.set_defaults(to_test_params=False)
    subparsers = parser.add_subparsers(help='There are two commands:\n'
                                            '- start <task_command> <task arguments>: starts a new task\n'
                                            '- continue: continues previously stopped task')

    continue_parser = subparsers.add_parser('continue')
    continue_parser.set_defaults(to_continue=True)

    start_task_parser = subparsers.add_parser('start', help='syntax: start <task command> <task arguments>\n'
                                                            'possible task commands:\n\n'
                                                            '- train / train <model name> <last epoch>. First '
                                                            'version starts training a new model. Second version '
                                                            'starts training of the model with given name after '
                                                            'given epoch.\n\n'
                                                            '- test <model name> <test epoch>: test model with given '
                                                            'name after given epoch.\n\n'
                                                            '- test_params: starts parameter testing through '
                                                            'sampling.\n\n'
                                                            'Notes: \n\t- all other parameters are specified through '
                                                            'source code.\n\t- model names should be given in quotes.')
    start_task_subparsers = start_task_parser.add_subparsers(help='start task subparsers help')

    train_task_parser = start_task_subparsers.add_parser('train')
    train_task_parser.set_defaults(to_train=True)
    train_task_parser.add_argument('model_name', type=str, nargs='?')
    train_task_parser.add_argument('last_epoch', type=int, nargs='?')

    test_task_parser = start_task_subparsers.add_parser('test')
    test_task_parser.set_defaults(to_test=True)
    test_task_parser.add_argument('model_name', type=str)
    test_task_parser.add_argument('test_epoch', type=int)

    test_params_parser = start_task_subparsers.add_parser('test_params')
    test_params_parser.set_defaults(to_test_params=True)

    args = parser.parse_args()

    return args


def start_train(to_continue, model_name='', last_epoch=0):
    import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default():
        if to_continue:
            with open('metadata/train_last_args.pkl', 'rb') as file:
                last_args = pickle.load(file)
            train(graph, **last_args)
        else:
            train(graph,
                  model_name=model_name,
                  last_epoch=last_epoch,
                  to_validate=True,
                  validate_start_epoch=7,
                  train_on_full_dataset=False)


def start_test(to_continue, model_name=None, test_epoch=None):
    if to_continue:
        with open('metadata/test_last_args.pkl', 'rb') as file:
            last_args = pickle.load(file)
        test(**last_args)
    else:
        test(model_name, test_epoch, 0)


def start_test_params(to_continue):
    if to_continue:
        with open('metadata/test_params_last_args.pkl', 'rb') as file:
            last_args = pickle.load(file)
        test_params(**last_args)
    else:
        test_params(
            epoch_cnt=15,
            try_cnt=5,
            params=NetworkParams(['lr_decay_freq'], [(1, 5)], [False], [False], 5))


if __name__ == '__main__':
    args = get_parser_args()

    if args.to_continue:
        with open('metadata/last_task_args.pkl', 'rb') as file:
            last_task_args = pickle.load(file)

        if last_task_args.to_train:
            start_train(True)
        elif last_task_args.to_test:
            start_test(True)
        elif last_task_args.to_test_params:
            start_test_params(True)
        else:
            raise Exception('Wrong last task arguments')
    else:
        with open('metadata/last_task_args.pkl', 'wb') as file:
            pickle.dump(args, file)

        if args.to_train:
            if args.model_name is not None and args.last_epoch is not None:
                start_train(False, args.model_name, args.last_epoch)
            elif args.model_name is None and args.last_epoch is None:
                start_train(False)
            elif args.model_name is not None and args.last_epoch is None:
                raise Exception('Besides name of the model, last epoch number is required to train')
        elif args.to_test:
            start_test(False, args.model_name, args.test_epoch)
        elif args.to_test_params:
            start_test_params(False)
        else:
            raise Exception('Wrong arguments')
