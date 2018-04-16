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


def test(model_name, test_epoch, to_log=True):
    _, _, _, func_args = inspect.getargvalues(inspect.currentframe())
    save_args(func_args)

    dataset = 3

    load_dataset('ag_news', dataset)

    print_log(to_log, 'dataset loaded')

    vdcnn = VDCNN()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(config=config) as sess:
        vdcnn.load(
            sess=sess,
            model_name=model_name,
            test_epoch=test_epoch
        )

        # vdcnn.load_old(
        #     sess=sess,
        #     model_name=model_name,
        #     test_epoch=test_epoch
        # )

        timer.start()
        print_log(to_log, 'start testing after epoch {}', str(test_epoch))

        accuracy = 0
        step = 0
        batch_size = 100
        for batch in batch_iterator(CURRENT_DATASET, dataset, batch_size, False):
            feed_dict = {
                vdcnn.network_input: batch[0],
                vdcnn.correct_labels: batch[1],
                vdcnn.keep_prob: 1,
                vdcnn.is_training: False
            }

            batch_accuracy = sess.run(vdcnn.accuracy, feed_dict=feed_dict)

            actual_batch_size = batch[0].shape[0]
            accuracy = (accuracy * batch_size * step + batch_accuracy * actual_batch_size) / \
                       (batch_size * step + actual_batch_size)

            step += 1
            if step % 10 == 0:
                print_log(to_log, '{}th batch done, current accuracy {}', step, accuracy)

        print_log(to_log, 'testing done in {}', timer.stop())
        print_log(to_log, 'test error: {}', (1 - accuracy))

    return accuracy
