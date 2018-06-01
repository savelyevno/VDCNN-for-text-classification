import tensorflow as tf
import numpy as np
import pickle
import csv

from consts import CURRENT_DATASET, INVERSE_ALPHABET_DICT, PADDING_TOKEN
from preprocess_data import load_dataset, LOADED_DATASETS, DATASET_CLASS_NAMES
from Timer import timer
from get_data import batch_iterator
from VDCNN import VDCNN
from misc import print_log


def run(model_name, test_epoch, dataset_type):
    load_dataset('ag_news', dataset_type)

    vdcnn = VDCNN()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(config=config) as sess:
        vdcnn.load(
            sess=sess,
            model_name=model_name,
            epoch=test_epoch
        )

        batch_size = 100
        batch_cnt = -1
        wrong_sample_ids = []
        results = []
        for batch in batch_iterator(CURRENT_DATASET, dataset_type, batch_size, False):
            batch_cnt += 1
            feed_dict = {
                vdcnn.network_input: batch[0],
                vdcnn.correct_labels: batch[1],
                vdcnn.keep_prob: 1,
                vdcnn.is_training: False
            }

            predictions = sess.run(vdcnn.predictions, feed_dict=feed_dict)

            n_classes = predictions.shape[1]

            for i in range(batch_size):
                exp_sum = sum(np.exp(x) for x in predictions[i])
                probs = np.array([np.exp(predictions[i, j]) / exp_sum for j in range(n_classes)])
                labels = batch[1][i]

                results.append((np.argmax(probs) + 1, np.max(probs)))

                if np.argmax(probs) == np.argmax(labels):
                    continue

                wrong_sample_ids.append(batch_cnt*batch_size + i)

        return results, wrong_sample_ids


def validate_samples(model_name, test_epoch, dataset_type):
    results, wrong_sample_ids = run(model_name, test_epoch, dataset_type)

    text_data = []
    for arr in LOADED_DATASETS['ag_news'][dataset_type][0]:
        text = ""
        for x in arr:
            if x == PADDING_TOKEN:
                break
            text += INVERSE_ALPHABET_DICT[x]
        text_data.append(text)

    data = []
    for i in range(len(results)):
        pred_class, confd = results[i]
        data.append({
            'confidence': round(confd, 3),
            'correct class': DATASET_CLASS_NAMES['ag_news'][np.argmax(LOADED_DATASETS['ag_news'][dataset_type][1][i]) + 1],
            'text': str(text_data[i]),
            'predicted class': DATASET_CLASS_NAMES['ag_news'][pred_class]
        })

    wrong_data = []
    for i in wrong_sample_ids:
        pred_class, confd = results[i]
        wrong_data.append({
            'confidence': round(confd, 3),
            'correct class': DATASET_CLASS_NAMES['ag_news'][
                np.argmax(LOADED_DATASETS['ag_news'][dataset_type][1][i]) + 1],
            'text': str(text_data[i]),
            'predicted class': DATASET_CLASS_NAMES['ag_news'][pred_class]
        })
    wrong_data = list(sorted(wrong_data, key=lambda x: -x['confidence']))
    data = list(sorted(data, key=lambda x: -x['confidence']))

    with open('all_samples_' + model_name + '.csv', 'w') as file:
        fieldnames = ['confidence', 'predicted class', 'correct class', 'text']
        writer = csv.DictWriter(file, fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)

    with open('wrong_samples_' + model_name + '.csv', 'w') as file:
        fieldnames = ['confidence', 'predicted class', 'correct class', 'text']
        writer = csv.DictWriter(file, fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(wrong_data)


if __name__ == '__main__':
    validate_samples('2018-05-19 15:18:04', 14, 4)
