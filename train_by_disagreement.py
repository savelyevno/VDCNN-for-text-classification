import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import datetime
import inspect
import pickle
import numpy as np

from preprocess_data import load_datasets, LOADED_DATASETS
from consts import CURRENT_DATASET, TEXT_SIZE, DATASET_NCLASSES
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


def train_by_disagreement(
        graph,
        last_epoch=0,
        model_name_prefix='',
        model_name='',
        epoch_cnt=50,
        batch_size=128,
        initial_lr=1e-2,
        lr_after_init=1e-3,
        lr_decay_freq=2,
        to_save_all=True,
        to_save_last=False,
        to_log=True,
        to_load_dataset=True,
        # vdcnn=None,
        to_validate=False,
        train_on_full_dataset=False,
        validate_start_epoch=10,
        init_epoch_cnt=8):

    if to_load_dataset:
        load_datasets(CURRENT_DATASET)
        print_log(to_log, 'datasets loaded')

    if train_on_full_dataset:
        train_dataset = 1
    else:
        train_dataset = 2

    epoch_steps_cnt = LOADED_DATASETS[CURRENT_DATASET][train_dataset][0].shape[0] / batch_size

    global_step_vars = []
    vdcnns = []
    with tf.variable_scope('net1'):
        vdcnns.append(VDCNN())
        vdcnns[0].build()

        global_step_vars.append(tf.Variable(
            initial_value=0,
            trainable=False,
            dtype=tf.int32,
            name='global_step'))
    with tf.variable_scope('net2'):
        vdcnns.append(VDCNN())
        vdcnns[1].build()

        global_step_vars.append(tf.Variable(
            initial_value=0,
            trainable=False,
            dtype=tf.int32,
            name='global_step'))

    to_update = tf.stop_gradient(tf.to_float(tf.logical_or(
        x=tf.not_equal(vdcnns[0].predicted_classes, vdcnns[1].predicted_classes),
        y=tf.less(
            x=tf.to_float(global_step_vars[0]),
            y=init_epoch_cnt*epoch_steps_cnt))))
    to_update_ratio = tf.reduce_mean(to_update)

    train_steps = []

    with tf.variable_scope('net1'):
        loss_to_update1 = tf.div(tf.reduce_sum((vdcnns[0].cross_entropy + vdcnns[0].reg_loss) * to_update),
                                 tf.maximum(batch_size * 0.1, tf.reduce_sum(to_update)), name='loss_to_update')

        # learning_rate1 = tf.train.exponential_decay(
        #     learning_rate=vdcnns[0].learn_rate,
        #     global_step=global_step_vars[0],
        #     decay_steps=vdcnns[0].lr_decay_freq * epoch_steps_cnt,
        #     decay_rate=vdcnns[0].lr_decay_rate,
        #     staircase=True,
        #     name='learning_rate')
        learning_rate1_var = tf.Variable(initial_lr, trainable=False, name='learning_rate_var')
        learning_rate1 = tf.convert_to_tensor(learning_rate1_var, name='learning_rate')

        train_steps.append(tf.contrib.layers.optimize_loss(
            loss=loss_to_update1,
            global_step=global_step_vars[0],
            learning_rate=learning_rate1,
            optimizer='Adam',
            summaries=['gradients']))
        
    with tf.variable_scope('net2'):
        loss_to_update2 = tf.div(tf.reduce_sum((vdcnns[1].cross_entropy + vdcnns[1].reg_loss) * to_update),
                                 tf.maximum(batch_size * 0.1, tf.reduce_sum(to_update)), name='loss_to_update')

        # learning_rate2 = tf.train.exponential_decay(
        #     learning_rate=vdcnns[1].learn_rate,
        #     global_step=global_step_vars[1],
        #     decay_steps=vdcnns[1].lr_decay_freq * epoch_steps_cnt,
        #     decay_rate=vdcnns[1].lr_decay_rate,
        #     staircase=True,
        #     name='learning_rate')
        learning_rate2_var = tf.Variable(initial_lr, trainable=False, name='learning_rate_var')
        learning_rate2 = tf.convert_to_tensor(learning_rate2_var, name='learning_rate')

        train_steps.append(tf.contrib.layers.optimize_loss(
            loss=loss_to_update2,
            global_step=global_step_vars[1],
            learning_rate=learning_rate2,
            optimizer='Adam',
            summaries=['gradients']))

    average_loss = (vdcnns[0].loss + vdcnns[1].loss) / 2
    average_reg_loss = (vdcnns[0].reg_loss + vdcnns[1].reg_loss) / 2
    average_accuracy = (vdcnns[0].accuracy + vdcnns[1].accuracy) / 2

    print_log(to_log, 'network built')

    #
    # summaries
    #

    # Regular summary; is written every 100th batch
    tf.summary.scalar(name='average loss', tensor=average_loss)
    tf.summary.scalar(name='average accuracy', tensor=average_accuracy)
    tf.summary.scalar(name='average reg loss', tensor=average_reg_loss)

    tf.summary.scalar(name='loss difference', tensor=vdcnns[0].loss - vdcnns[1].loss)
    tf.summary.scalar(name='train acc difference', tensor=vdcnns[0].accuracy - vdcnns[1].accuracy)
    tf.summary.scalar(name='reg loss difference', tensor=vdcnns[0].reg_loss - vdcnns[1].reg_loss)

    tf.summary.scalar(name='batch diff ratio', tensor=to_update_ratio)

    summary = tf.summary.merge_all()

    # Validation summary; is written every 100th batch
    validation_accuracy_placeholders = []
    validation_summaries = []
    for i_vdcnn in range(2):
        validation_accuracy_placeholders.append(tf.placeholder(tf.float32))
        validation_summaries.append(tf.summary.scalar(
            name='vld1_accuracy' + str(i_vdcnn + 1),
            tensor=validation_accuracy_placeholders[i_vdcnn]))

    # Test summary; is written after every epoch
    test_accuracy_placeholders = []
    test_summaries = []
    for i_vdcnn in range(2):
        test_accuracy_placeholders.append(tf.placeholder(tf.float32))
        test_summaries.append(tf.summary.scalar(
            name='test_accuracy' + str(i_vdcnn + 1),
            tensor=test_accuracy_placeholders[i_vdcnn]))

    saver = tf.train.Saver(max_to_keep=0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
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
        best_vld1_accuracy = 0
        best_vld2_accuracy = 0
        timer.start()

        print_log(to_log, 'start training...')

        for i_epoch in range(last_epoch + 1, last_epoch + epoch_cnt + 1):
            timer.start()
            print_log(to_log, '\nstart at epoch {}', i_epoch)

            #
            # update lr by hand
            #
            # if i_epoch <= init_epoch_cnt:
            #     lr_decay_cnt = (i_epoch - 1)//lr_decay_freq
            #     new_lr = initial_lr / 2 ** lr_decay_cnt
            # else:
            #     lr_decay_cnt = (i_epoch - init_epoch_cnt - 1) // lr_decay_freq
            #     new_lr = lr_after_init / 2 ** lr_decay_cnt
            lr_decay_cnt = (i_epoch - 1) // lr_decay_freq
            new_lr = initial_lr / 2 ** lr_decay_cnt

            lr1_upd_op = learning_rate1_var.assign(new_lr)
            sess.run(lr1_upd_op)

            lr2_upd_op = learning_rate2_var.assign(new_lr)
            sess.run(lr2_upd_op)

            # if i_epoch <= init_epoch_cnt:
            #     batch_seq1 = list(batch_iterator(CURRENT_DATASET, 5, batch_size))
            #     batch_seq2 = list(batch_iterator(CURRENT_DATASET, 6, batch_size))
            #     assert len(batch_seq1) == len(batch_seq2)
            #     batches_cnt = len(batch_seq1)
            # else:
            batch_seq = list(batch_iterator(CURRENT_DATASET, train_dataset, batch_size))
            batches_cnt = len(batch_seq)

            timer.start()
            for batch_i in range(batches_cnt):
                feed_dict = {}
                for i in range(2):
                    vdcnn = vdcnns[i]
                    if i_epoch <= init_epoch_cnt:
                        # batch1 = batch_seq1[batch_i]
                        # batch2 = batch_seq2[batch_i]
                        batch1 = batch_seq[batch_i]
                        batch2 = batch_seq[batch_i]
                        if i == 0:
                            feed_dict[vdcnn.network_input] = batch1[0]
                            feed_dict[vdcnn.correct_labels] = batch1[1]
                            feed_dict[vdcnn.keep_prob] = 0.5
                            feed_dict[vdcnn.is_training] = True
                        else:
                            feed_dict[vdcnn.network_input] = batch2[0]
                            feed_dict[vdcnn.correct_labels] = batch2[1]
                            feed_dict[vdcnn.keep_prob] = 0.5
                            feed_dict[vdcnn.is_training] = True
                    else:
                        batch = batch_seq[batch_i]
                        feed_dict[vdcnn.network_input] = batch[0]
                        feed_dict[vdcnn.correct_labels] = batch[1]
                        feed_dict[vdcnn.keep_prob] = 0.5
                        feed_dict[vdcnn.is_training] = True

                if (global_step + 1) % 100 != 0:
                    _, _, _, global_step = sess.run([
                            train_steps[0],
                            train_steps[1],
                            global_step_vars[0],
                            global_step_vars[1]],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * batch_size
                else:
                    _, _, _, global_step, loss, train_accuracy, summary_str, lr, reg_loss = sess.run(
                        [
                            train_steps[0],
                            train_steps[1],
                            global_step_vars[0],
                            global_step_vars[1],
                            average_loss,
                            average_accuracy,
                            summary,
                            learning_rate1,
                            average_reg_loss],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * batch_size

                    print_log(to_log,
                              '\ttrain samples cnt: {}, avg. train acc. {}, avg. loss {}, avg. reg loss {}, lr: {}; dt = {}',
                              train_samples_cnt,
                              str(round(train_accuracy, 3)),
                              str(round(loss, 3)),
                              str(round(reg_loss, 5)),
                              str(round(lr, 10)),
                              timer.stop_start())

                    # Write summary
                    if to_save_all or to_save_last:
                        summary_writer.add_summary(summary_str, train_samples_cnt)
                        summary_writer.flush()

                if to_validate and i_epoch >= validate_start_epoch:
                    for i_vdcnn in range(2):
                        vdcnn = vdcnns[i_vdcnn]

                        feed_dict[vdcnn.is_training] = False
                        vld1_accuracy = calc_accuracy(sess=sess,
                                                      accuracy=vdcnn.accuracy,
                                                      input_tensor=vdcnn.network_input,
                                                      y_=vdcnn.correct_labels,
                                                      feed_dict=feed_dict,
                                                      data_set_type=5)
                        # Write summary
                        summary_str = sess.run(validation_summaries[i_vdcnn], feed_dict={validation_accuracy_placeholders[i_vdcnn]: vld1_accuracy})
                        summary_writer.add_summary(summary_str, train_samples_cnt + 1)
                        summary_writer.flush()

                        if (vld1_accuracy - best_vld1_accuracy) > -1e-2:
                            if vld1_accuracy > best_vld1_accuracy:
                                best_vld1_accuracy = vld1_accuracy

                            vld2_accuracy = calc_accuracy(
                                sess=sess,
                                accuracy=vdcnn.accuracy,
                                input_tensor=vdcnn.network_input,
                                y_=vdcnn.correct_labels,
                                feed_dict=feed_dict,
                                data_set_type=4)

                            summary_str = sess.run(test_summaries[i_vdcnn], feed_dict={test_accuracy_placeholders[i_vdcnn]: vld2_accuracy})
                            summary_writer.add_summary(summary_str, train_samples_cnt + 2)
                            summary_writer.flush()

                            if vld2_accuracy > best_vld2_accuracy:
                                best_vld2_accuracy = vld2_accuracy

                                print_log(to_log, '\tbest vld2 accuracy update for net{}: {}', i_vdcnn + 1, vld2_accuracy)

                                saver.save(
                                    sess=sess,
                                    save_path=str(checkpoint_folder_path) + '/model_best',
                                    global_step=i_epoch,
                                    write_meta_graph=False)

            for i_vdcnn in range(2):
                vdcnn = vdcnns[i_vdcnn]

                feed_dict[vdcnn.is_training] = False
                vld_accuracy = calc_accuracy(
                    sess=sess,
                    accuracy=vdcnn.accuracy,
                    input_tensor=vdcnn.network_input,
                    y_=vdcnn.correct_labels,
                    feed_dict=feed_dict,
                    data_set_type=4)
                best_accuracy = max(best_accuracy, vld_accuracy)
                print_log(to_log, 'validation{} accuracy: {}', i_vdcnn + 1, vld_accuracy)

                summary_str = sess.run(validation_summaries[i_vdcnn], feed_dict={validation_accuracy_placeholders[i_vdcnn]: vld_accuracy})
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
