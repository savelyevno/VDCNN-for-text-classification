import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import datetime
import inspect
import pickle

from preprocess_data import load_datasets, LOADED_DATASETS
from consts import CURRENT_DATASET
from Timer import timer
from get_data import batch_iterator
from VDCNN import VDCNN
from misc import print_log


def load_embeddings_graph(summary_writer):
    # https://stackoverflow.com/questions/42679552/how-can-i-select-which-checkpoint-to-view-in-tensorboards-embeddings-tab
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedding'
    embedding.metadata_path = '/home/nikita/PycharmProjects/VDCNN-for-text-classification/checkpoints/embeddings_metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)


def load_checkpoint_path(model_name):
    checkpoint_folder = 'checkpoints/' + model_name + '/'

    checkpoint_folder_path = os.path.abspath(os.path.join(os.path.curdir, checkpoint_folder))
    if not os.path.exists(checkpoint_folder_path):
        os.makedirs(checkpoint_folder_path)

    return checkpoint_folder_path


def calc_accuracy(sess, accuracy, input_tensor, y_, feed_dict, data_set_type):
    sm = 0
    cnt = 0
    for batch in batch_iterator(CURRENT_DATASET, data_set_type, 128, False):
        feed_dict[input_tensor] = batch[0]
        feed_dict[y_] = batch[1]
        batch_acc = sess.run(accuracy, feed_dict=feed_dict)

        batch_size = batch[0].shape[0]
        cnt += batch_size
        sm += batch_acc * batch_size
    acc = sm / cnt

    return acc


def train(
        graph,
        last_epoch=0,
        model_name_prefix='',
        model_name='',
        epoch_cnt=50,
        batch_size=128,
        to_save_all=True,
        to_save_last=False,
        to_log=True,
        to_load_dataset=True,
        vdcnn=None,
        to_validate=False,
        train_on_full_dataset=False,
        validate_start_epoch=10):

    if to_load_dataset:
        load_datasets(CURRENT_DATASET)
        print_log(to_log, 'datasets loaded')

    if train_on_full_dataset:
        train_dataset = 1
    else:
        train_dataset = 2

    if vdcnn is None:
        vdcnn = VDCNN()
        vdcnn.build()

    global_step = tf.Variable(
        initial_value=0,
        trainable=False,
        dtype=tf.int32,
        name='global_step')
    with tf.name_scope('adam_optimizer'):
        learning_rate = tf.train.exponential_decay(
            learning_rate=vdcnn.learn_rate,
            global_step=global_step,
            decay_steps=vdcnn.lr_decay_freq*(LOADED_DATASETS[CURRENT_DATASET][train_dataset][0].shape[0] / batch_size),
            decay_rate=vdcnn.lr_decay_rate,
            staircase=True,
            name='learning_rate')

        train_step = tf.contrib.layers.optimize_loss(
            loss=vdcnn.loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            summaries=['gradients'])

    print_log(to_log, 'network built')

    #
    # summaries
    #

    # Regular summary; is written every 100th batch
    tf.summary.scalar(name='loss', tensor=vdcnn.loss)
    tf.summary.scalar(name='train_accuracy', tensor=vdcnn.accuracy)
    tf.summary.scalar(name='reg loss', tensor=vdcnn.reg_loss)
    summary = tf.summary.merge_all()

    # Validation summary; is written every 100th batch
    validation_accuracy_placeholder = tf.placeholder(tf.float32)
    validation_summary = tf.summary.scalar(
        name='validation_accuracy_new',
        tensor=validation_accuracy_placeholder)

    # Test summary; is written after every epoch
    test_accuracy_placeholder = tf.placeholder(tf.float32)
    test_summary = tf.summary.scalar(
        name='test_accuracy',
        tensor=test_accuracy_placeholder)

    saver = tf.train.Saver(max_to_keep=0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(graph=graph, config=config) as sess:
        # We start a new evaluation in case last_epoch=0 and load previous otherwise
        if last_epoch != 0:
            saver.restore(sess, 'checkpoints/' + model_name_prefix + model_name + '/model-' + str(last_epoch))
        else:
            model_name = str(datetime.datetime.now())[:-7]
            sess.run(tf.global_variables_initializer())

        if to_save_all or to_save_last:
            checkpoint_folder_path = load_checkpoint_path(model_name_prefix + model_name)
            summary_writer = tf.summary.FileWriter('checkpoints/' + model_name_prefix + model_name + '/', sess.graph)

            if last_epoch == 0:
                saver.save(sess, str(checkpoint_folder_path) + '/model', global_step=0)
                load_embeddings_graph(summary_writer)

        global_step = 0
        train_samples_cnt = 0
        best_accuracy = 0
        best_validation_accuracy = 0
        timer.start()

        print_log(to_log, 'start training...')

        for i_epoch in range(last_epoch + 1, last_epoch + epoch_cnt + 1):
            timer.start()
            print_log(to_log, '\nstart at epoch {}', i_epoch)

            timer.start()
            for batch in batch_iterator(CURRENT_DATASET, train_dataset, batch_size):
                feed_dict = {
                    vdcnn.network_input: batch[0],
                    vdcnn.correct_labels: batch[1],
                    vdcnn.keep_prob: 0.5,
                    vdcnn.is_training: True
                }

                if (global_step + 1) % 100 != 0:
                    _, global_step, train_accuracy = sess.run(
                        [
                            train_step,
                            global_step,
                            vdcnn.accuracy],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * batch_size
                else:
                    _, global_step, loss, train_accuracy, summary_str, lr, reg_loss = sess.run(
                        [
                            train_step,
                            global_step,
                            vdcnn.loss,
                            vdcnn.accuracy,
                            summary,
                            learning_rate,
                            vdcnn.reg_loss],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * batch_size

                    print_log(to_log,
                              '\ttrain samples cnt: {}, train accuracy {}, loss {}, reg loss {}, lr: {}; dt = {}',
                              train_samples_cnt,
                              train_accuracy,
                              loss,
                              reg_loss,
                              lr,
                              timer.stop_start())

                    # Write summary
                    if to_save_all or to_save_last:
                        summary_writer.add_summary(summary_str, train_samples_cnt)
                        summary_writer.flush()

                if to_validate and i_epoch >= validate_start_epoch:
                    feed_dict[vdcnn.is_training] = False
                    validation_accuracy = calc_accuracy(sess=sess,
                                                        accuracy=vdcnn.accuracy,
                                                        input_tensor=vdcnn.network_input,
                                                        y_=vdcnn.correct_labels,
                                                        feed_dict=feed_dict,
                                                        data_set_type=3)
                    # Write summary
                    summary_str = sess.run(validation_summary, feed_dict={validation_accuracy_placeholder: validation_accuracy})
                    summary_writer.add_summary(summary_str, train_samples_cnt + 1)
                    summary_writer.flush()

                    if (validation_accuracy - best_validation_accuracy) > -1e-2:
                        if validation_accuracy > best_validation_accuracy:
                            best_validation_accuracy = validation_accuracy

                        test_accuracy = calc_accuracy(
                            sess=sess,
                            accuracy=vdcnn.accuracy,
                            input_tensor=vdcnn.network_input,
                            y_=vdcnn.correct_labels,
                            feed_dict=feed_dict,
                            data_set_type=0)

                        summary_str = sess.run(test_summary, feed_dict={test_accuracy_placeholder: test_accuracy})
                        summary_writer.add_summary(summary_str, train_samples_cnt + 2)
                        summary_writer.flush()

                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy

                            print_log(to_log, '\tbest test accuracy update: {}', test_accuracy)

                            saver.save(
                                sess=sess,
                                save_path=str(checkpoint_folder_path) + '/model_best',
                                global_step=i_epoch,
                                write_meta_graph=False)

            feed_dict[vdcnn.is_training] = False
            test_accuracy = calc_accuracy(
                sess=sess,
                accuracy=vdcnn.accuracy,
                input_tensor=vdcnn.network_input,
                y_=vdcnn.correct_labels,
                feed_dict=feed_dict,
                data_set_type=0)
            best_accuracy = max(best_accuracy, test_accuracy)
            print_log(to_log, 'test accuracy: {}', test_accuracy)

            summary_str = sess.run(test_summary, feed_dict={test_accuracy_placeholder: test_accuracy})
            summary_writer.add_summary(summary_str, train_samples_cnt + 3)
            summary_writer.flush()

            if to_save_all or (to_save_last and i_epoch == last_epoch + epoch_cnt):
                saver.save(
                    sess=sess,
                    save_path=str(checkpoint_folder_path) + '/model',
                    global_step=i_epoch,
                    write_meta_graph=False)

            timer.stop()
            print_log(to_log, '\t{}th epoch done in {}', i_epoch, timer.stop())

        print_log(to_log, 'training done in {}', timer.stop())

    return best_accuracy, model_name
