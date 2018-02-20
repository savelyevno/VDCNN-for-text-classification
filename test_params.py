import tensorflow as tf
import inspect
import pickle

from preprocess_data import load_datasets
from consts import CURRENT_DATASET
from misc import print_log, Print
from VDCNN import VDCNN
from train import train


def save_args(args):
    with open('metadata/test_params_last_args.pkl', 'wb') as file:
        pickle.dump(args, file)


def test_params(
        epoch_cnt,
        try_cnt,
        params,
        start_i=0,
        results=None
        ):
    _, _, _, func_args = inspect.getargvalues(inspect.currentframe())
    save_args(func_args)

    if results is None:
        results = []
        func_args['results'] = results

    load_datasets(CURRENT_DATASET)
    print_log(True, 'dataset loaded')

    vdcnn = VDCNN(data_set_name=CURRENT_DATASET)

    for i in range(start_i, try_cnt):
        params.sample()

        graph = tf.Graph()
        with graph.as_default():

            vdcnn.set_params(params)
            vdcnn.build()

            accuracy = train(
                epoch_cnt=epoch_cnt,
                to_save=False,
                to_save_args=False,
                to_log=False,
                to_load_dataset=False,
                vdcnn=vdcnn,
                graph=graph)

        results.append((accuracy, str(params)))

        Print('{}: accuracy={}; params={}', i, accuracy, str(params))

        func_args['start_i'] = i + 1
        func_args['results'] = results
        save_args(func_args)

    results.sort(key=lambda tup: tup[0])

    for acc, s in results:
        Print('{}\t' + s, acc)
