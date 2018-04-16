import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import datetime
import inspect
import pickle

from preprocess_data import load_datasets
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
    embedding.metadata_path = os.path.abspath(os.path.join(os.path.curdir, 'checkpoints/embeddings_metadata.tsv'))
    projector.visualize_embeddings(summary_writer, config)


def load_checkpoint_path(model_name):
    checkpoint_folder = 'checkpoints/' + model_name + '/'

    checkpoint_folder_path = os.path.abspath(os.path.join(os.path.curdir, checkpoint_folder))
    if not os.path.exists(checkpoint_folder_path):
        os.makedirs(checkpoint_folder_path)

    return checkpoint_folder_path


def calc_accuracy(sess, accuracy, input_tensor, y_, feed_dict, data_set_type):
    acc = 0
    step = 0
    for batch in batch_iterator(CURRENT_DATASET, data_set_type, 128, False):
        batch_size = batch[0].shape[0]

        feed_dict[input_tensor] = batch[0]
        feed_dict[y_] = batch[1]
        batch_val_acc = sess.run(accuracy, feed_dict=feed_dict)

        # Updating average accuracy
        acc = (acc * step * 128 + batch_val_acc * batch_size) / (step * 128 + batch_size)
        step += 1

    return acc


def save_args(args):
    with open('metadata/train_last_args.pkl', 'wb') as file:
        pickle.dump(args, file)


def train(
        graph,
        last_epoch=0,
        model_name_prefix='',
        model_name='',
        epoch_cnt=50,
        to_save_all=True,
        to_save_last=False,
        to_log=True,
        to_load_dataset=True,
        vdcnn=None,
        to_save_args=True,
        to_validate=False):

    if to_save_args:
        _, _, _, func_args = inspect.getargvalues(inspect.currentframe())
        del func_args['graph']
        save_args(func_args)

    if vdcnn is None:
        vdcnn = VDCNN(data_set_name=CURRENT_DATASET)
        vdcnn.build()

    print_log(to_log, 'network built')

    # Regular summary; is written every 100th batch
    tf.summary.scalar(name='loss', tensor=vdcnn.loss)
    tf.summary.scalar(name='train_accuracy', tensor=vdcnn.accuracy)
    tf.summary.scalar(name='reg loss', tensor=vdcnn.reg_loss)
    summary = tf.summary.merge_all()

    # Validation summary; is written every 100th batch
    validation_accuracy_placeholder = tf.placeholder(tf.float32)
    validation_summary = tf.summary.scalar(
        name='validation_accuracy_new',
        tensor=validation_accuracy_placeholder,
        collections=['per_epoch'])

    # Test summary; is written after every epoch
    test_accuracy_placeholder = tf.placeholder(tf.float32)
    test_summary = tf.summary.scalar(
        name='test_accuracy',
        tensor=test_accuracy_placeholder,
        collections=['per_epoch'])

    saver = tf.train.Saver(max_to_keep=0)

    if to_load_dataset:
        load_datasets(CURRENT_DATASET)
        print_log(to_log, 'dataset loaded')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "1"
    with tf.Session(graph=graph, config=config) as sess:
        # We start a new evaluation in case last_epoch=0 and load previous otherwise

        if last_epoch != 0:
            saver.restore(sess, 'checkpoints/' + model_name_prefix + model_name + '/model-' + str(last_epoch))
        else:
            model_name = str(datetime.datetime.now())[:-7]
            sess.run(tf.global_variables_initializer())

            if to_save_args:
                func_args['model_name'] = model_name
                save_args(func_args)

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
        train_accuracy = 0
        timer.start()

        print_log(to_log, 'start training...')

        for i_epoch in range(last_epoch + 1, last_epoch + epoch_cnt + 1):
            timer.start()
            print_log(to_log, '\nstart at epoch {}', i_epoch)

            timer.start()
            for batch in batch_iterator(CURRENT_DATASET, 1, vdcnn.batch_size):
                feed_dict = {
                    vdcnn.network_input: batch[0],
                    vdcnn.correct_labels: batch[1],
                    vdcnn.keep_prob: 0.5,
                    vdcnn.is_training: True
                }

                if (global_step + 1) % 100 != 0:
                    _, global_step, train_accuracy = sess.run(
                        [
                            vdcnn.train_step,
                            vdcnn.global_step,
                            vdcnn.accuracy],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * vdcnn.batch_size
                else:
                    _, global_step, loss, train_accuracy, summary_str, lr, reg_loss = sess.run(
                        [
                            vdcnn.train_step,
                            vdcnn.global_step,
                            vdcnn.loss,
                            vdcnn.accuracy,
                            summary,
                            vdcnn.learning_rate,
                            vdcnn.reg_loss],
                        feed_dict=feed_dict)
                    train_samples_cnt = global_step * vdcnn.batch_size

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

                if to_validate and i_epoch > 18:
                    feed_dict[vdcnn.is_training] = False
                    validation_accuracy = calc_accuracy(sess=sess,
                                                        accuracy=vdcnn.accuracy,
                                                        input_tensor=vdcnn.network_input,
                                                        y_=vdcnn.correct_labels,
                                                        feed_dict=feed_dict,
                                                        data_set_type=2)
                    # Write summary
                    summary_str = sess.run(validation_summary, feed_dict={validation_accuracy_placeholder: validation_accuracy})
                    summary_writer.add_summary(summary_str, train_samples_cnt)
                    summary_writer.flush()

                    if (validation_accuracy - best_validation_accuracy) > -2e-2:
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
                        summary_writer.add_summary(summary_str, train_samples_cnt)
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
            summary_writer.add_summary(summary_str, train_samples_cnt)
            summary_writer.flush()

            if to_save_all or (to_save_last and i_epoch == last_epoch + epoch_cnt):
                saver.save(
                    sess=sess,
                    save_path=str(checkpoint_folder_path) + '/model',
                    global_step=i_epoch,
                    write_meta_graph=False)

            if to_save_args:
                func_args['last_epoch'] = i_epoch
                save_args(func_args)

            timer.stop()
            print_log(to_log, '\t{}th epoch done in {}', i_epoch, timer.stop())

        print_log(to_log, 'training done in {}', timer.stop())

    return best_accuracy, model_name
