import tensorflow as tf
import inspect
import pickle
import csv
import datetime
import copy

from preprocess_data import load_datasets
from consts import CURRENT_DATASET
from misc import print_log, Print
from VDCNN import VDCNN
from train import train


def save_args(args):
    with open('metadata/test_params_last_args.pkl', 'wb') as file:
        pickle.dump(args, file)


def append_to_csv(csv_file_name, fieldnames, next_params, accuracy):
    write_dict = copy.copy(next_params)
    write_dict['acc'] = accuracy
    with open(csv_file_name, 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writerow(write_dict)


def test_params(
        epoch_cnt,
        try_cnt,
        params,
        eval_name=str(datetime.datetime.now())[:-7],
        start_i=0,
        results=None,
        ):

    _, _, _, func_args = inspect.getargvalues(inspect.currentframe())
    save_args(func_args)

    if results is None:
        results = []
        func_args['results'] = results

    csv_fieldnames = copy.copy(params.names)
    csv_fieldnames.append('acc')
    csv_file_name = 'checkpoints/test_params/' + eval_name + '.csv'
    if start_i == 0:
        with open(csv_file_name, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            csv_writer.writeheader()

    load_datasets(CURRENT_DATASET)
    print_log(True, 'dataset loaded')

    vdcnn = VDCNN(data_set_name=CURRENT_DATASET)

    for i in range(start_i, try_cnt):
        graph = tf.Graph()
        with graph.as_default():
            next_params = params.get_next()
            vdcnn.set_params(next_params)
            vdcnn.build()

            accuracy, model_name = train(
                epoch_cnt=epoch_cnt,
                to_save_all=False,
                to_save_last=True,
                to_save_args=False,
                to_log=True,
                to_load_dataset=False,
                model_name_prefix='test_params/' + eval_name + '/',
                vdcnn=vdcnn,
                graph=graph)

        results.append((accuracy, model_name, str(next_params)))

        Print('{}: accuracy={}; model_name=\'{}\'; params={}', i, accuracy, model_name, str(next_params))
        append_to_csv(csv_file_name, csv_fieldnames, next_params, accuracy)

        func_args['start_i'] = i + 1
        func_args['results'] = results
        save_args(func_args)

    results.sort(key=lambda tup: tup[0])

    for acc, s, params in results:
        Print('{}\t {}, {}', acc, s, params)
