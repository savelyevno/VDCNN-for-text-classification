import numpy as np
import csv
import tensorflow as tf

from VDCNN import VDCNN
from preprocess_data import preprocess_text
from consts import TEXT_SIZE
from preprocess_data import LOADED_DATASETS

class_ids = {
    'World': 0,
    'Sports': 1,
    'Business': 2,
    'Sci/Tech': 3,
    '?????': 4
}
class_names = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech',
    4: '?????'
}


# works only for ag_news
def load_custom_dataset(dict_rows):

    N = len(dict_rows)

    X = np.empty((N, TEXT_SIZE), dtype=np.int32)
    for i in range(N):
        text = dict_rows[i]['text']

        X[i, :] = preprocess_text(text)

    return X


def run_on_custom_dataset_with_disagreement(model_name, epoch):
    with open('my_labels.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')

        dict_rows = [row for row in reader]

    X = load_custom_dataset(dict_rows)

    vdcnn = VDCNN()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(config=config) as sess:
        results = [[], []]
        for i in range(2):
            vdcnn.load(
                sess=sess,
                model_name=model_name,
                epoch=epoch,
                var_scope='net{}/'.format(i + 1)
            )

            feed_dict = {
                vdcnn.network_input: X,
                vdcnn.keep_prob: 1,
                vdcnn.is_training: False
            }

            results[i] = sess.run(vdcnn.predicted_classes, feed_dict)

    N = len(dict_rows)
    for i_vdcnn in range(2):
        sum_my = 0
        cnt_my = 0
        sum_cor = 0
        cnt_cor = 0
        for i in range(N):
            row = dict_rows[i]

            if class_ids[row['my class']] != 4:
                sum_my += int(results[i_vdcnn][i] == class_ids[row['my class']])
                cnt_my += 1

                sum_cor += int(results[i_vdcnn][i] == class_ids[row['correct class']])
                cnt_cor += 1

        acc_my = sum_my/cnt_my
        acc_cor = sum_cor/N

        print('{}: unlabeled data: {}/{}'.format(i_vdcnn, N - cnt_my, N))
        print('{}: my acc: {}; cor acc: {}'.format(i_vdcnn, acc_my, acc_cor))


def run_on_custom_dataset(model_name, epoch):
    with open('my_labels.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')

        dict_rows = [row for row in reader]

    X = load_custom_dataset(dict_rows)

    vdcnn = VDCNN()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(config=config) as sess:
        vdcnn.load(
            sess=sess,
            model_name=model_name,
            epoch=epoch
        )
        feed_dict = {
            vdcnn.network_input: X,
            vdcnn.keep_prob: 1,
            vdcnn.is_training: False
        }
        results = sess.run(vdcnn.predictions, feed_dict)
        results = np.argmax(results, axis=1)

    N = len(dict_rows)
    sum_my = 0
    cnt_my = 0
    sum_cor = 0
    cnt_cor = 0
    for i in range(N):
        row = dict_rows[i]
        if class_ids[row['my class']] != 4:
            sum_my += int(results[i] == class_ids[row['my class']])
            cnt_my += 1
            sum_cor += int(results[i] == class_ids[row['correct class']])
            cnt_cor += 1
    acc_my = sum_my/cnt_my
    acc_cor = sum_cor/cnt_cor

    print('unlabeled data: {}/{}'.format(N - cnt_my, N))
    print('my acc: {}; cor acc: {}'.format(acc_my, acc_cor))


if __name__ == '__main__':
    # run_on_custom_dataset_with_disagreement('2018-04-29 21:26:43', 22)      # 0.5947
    # run_on_custom_dataset_with_disagreement('2018-04-30 13:13:13', 16)      # 0.5797
    # run_on_custom_dataset_with_disagreement('2018-04-30 20:49:42', 13)      # 0.5849
    # run_on_custom_dataset_with_disagreement('2018-04-30 20:49:42', 18)      # 0.5947
    # run_on_custom_dataset_with_disagreement('2018-05-01 09:41:14', 15)      # 0.5947
    # run_on_custom_dataset_with_disagreement('2018-05-02 12:17:04', 17)      # 0.598
    # run_on_custom_dataset_with_disagreement('2018-05-03 13:50:56', 14)      # 0.598

    # for i in range(14, 20):     # 0.62, 0.59, 0.61, 0.6, 0.6, 0.59
    #     run_on_custom_dataset_with_disagreement('2018-05-02 12:17:04', i)

    # run_on_custom_dataset('2018-04-22 17:03:20', 12)                        # 0.562
    # run_on_custom_dataset('2018-04-16 15:23:21', 13)                        # 0.562

    # trained on full dataset
    # run_on_custom_dataset('2018-04-17 12:51:14', 7)                         # 0.5686
    # run_on_custom_dataset('2018-04-17 12:51:14', 11)                        # 0.5621
