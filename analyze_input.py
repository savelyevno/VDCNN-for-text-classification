import tensorflow as tf
import numpy as np

from preprocess_data import preprocess_text
from VDCNN import VDCNN

model_name = '2018-04-17 12:51:14'
epoch = 11


def foo(texts):
    data = np.array([preprocess_text(text) for text in texts])

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
            vdcnn.network_input: data,
            vdcnn.keep_prob: 1,
            vdcnn.is_training: False
        }

        results = sess.run(vdcnn.predictions, feed_dict)

    for i in range(len(texts)):
        result = results[i]
        print(texts[i])
        # print('\t {}'.format(sorted(result, key=lambda x: -x)[:20]))
        print('\t {}'.format(result))
        print('\t max: ', round(np.max(result), 3))
        print('\t argmax: ', np.argmax(result))
        print('\t mean: ', round(np.mean(result), 3))
        print('\t median: ', round(np.median(result), 3))
        print()


if __name__ == '__main__':
    foo(['sport',
         'soccer',
         'dfsgn',
         'bghdgnfa',
         'the',
         'income',
         'value',
         'iraq'])
