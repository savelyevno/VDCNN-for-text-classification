import tensorflow as tf
import pickle
import inspect

from consts import CURRENT_DATASET
from preprocess_data import load_dataset
from Timer import timer
from get_data import batch_iterator
from VDCNN import VDCNN
from misc import print_log


def save_args(args):
    with open('metadata/test_last_args.pkl', 'wb') as file:
        pickle.dump(args, file)


def test(model_name, test_epochs, datasets, to_log=True, model_id=None):
    _, _, _, func_args = inspect.getargvalues(inspect.currentframe())
    save_args(func_args)

    for dataset in datasets:
        load_dataset(CURRENT_DATASET, dataset)

    print_log(to_log, 'dataset loaded')

    vdcnn = VDCNN()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        for test_epoch in test_epochs:
            print_log(to_log, 'start testing after epoch {}', test_epoch)
            for dataset in datasets:
                print_log(to_log, '\ton dataset {}', dataset)
                vdcnn.load(
                    sess=sess,
                    model_name=model_name,
                    epoch=test_epoch
                )

                cnt = 0
                sm = 0
                for batch in batch_iterator(CURRENT_DATASET, dataset, 128, False):
                    feed_dict = {
                        vdcnn.network_input: batch[0],
                        vdcnn.correct_labels: batch[1],
                        vdcnn.keep_prob: 1,
                        vdcnn.is_training: False
                    }

                    batch_accuracy = sess.run(vdcnn.accuracy, feed_dict=feed_dict)

                    batch_size = batch[0].shape[0]
                    cnt += batch_size
                    sm += batch_size * batch_accuracy
                acc = sm / cnt

                print_log(to_log, '\taccuracy: {}; error: {}', str(round(acc, 5)), str(round((1 - acc), 5)))
