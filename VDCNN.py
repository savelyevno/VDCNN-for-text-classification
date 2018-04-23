import tensorflow as tf
from consts import TEXT_SIZE, EMBEDDING_SIZE, CURRENT_DATASET, DATASET_NCLASSES, ALPHABET_DICT, EMBEDDING_GEN_SEED, \
    DATASET_SIZE
import numpy as np
from tensorflow.python.ops import control_flow_ops


class VDCNN:
    def __init__(self,
                 max_st_dev=1.0,
                 # max_st_dev=1e-2,
                 reg_coef=1e-4,
                 # reg_coef=0.0,
                 learn_rate=9e-3,
                 # learn_rate=1e-2,
                 lr_decay_rate=0.5,
                 lr_decay_freq=2,
                 batch_size=128,
                 embedding_size=16,
                 feature_cnts=[64, 128, 256, 512],
                 # conv_block_cnts=[1, 1, 1, 1],      # 9 convolutional layers
                 # conv_block_cnts=[2, 2, 2, 2],      # 17 convolutional layers
                 conv_block_cnts=[2, 2, 5, 5],      # 29 convolutional layers
                 hidden_layers_cnt=2048,
                 k_max_pool_cnt=8,
                 use_dropout=False,
                 data_set_name=CURRENT_DATASET):

        self.max_st_dev = max_st_dev
        self.reg_coef = reg_coef
        self.learn_rate = learn_rate
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.feature_cnts = feature_cnts
        self.conv_block_cnts = conv_block_cnts
        self.hidden_layers_cnt = hidden_layers_cnt
        self.k_max_pool_cnt = k_max_pool_cnt
        self.use_dropout = use_dropout
        self.data_set_name = data_set_name

        self.params_cnt = 0

    @staticmethod
    def _conv1d(input_tnsr, filter_weights):
        return tf.nn.conv2d(
            input=input_tnsr,
            filter=filter_weights,
            strides=[1, 1, 1, 1],
            padding='SAME')

    def _weight_variable(self, shape, st_dev):
        prod = 1
        for dim in shape:
            prod *= dim
        self.params_cnt += prod

        return tf.Variable(
            tf.truncated_normal(
                shape=shape,
                stddev=min(self.max_st_dev, st_dev))
        )

    def _bias_variable(self, shape):
        prod = 1
        for dim in shape:
            prod *= dim
        self.params_cnt += prod

        return tf.Variable(tf.constant(value=0.0, shape=shape))

    def _batch_norm(self, input_tnsr, is_training):
        with tf.variable_scope('batch_norm'):
            phase_train = tf.convert_to_tensor(is_training, dtype=tf.bool)

            if len(input_tnsr.get_shape()) > 2:
                n_out = int(input_tnsr.get_shape()[3])
            else:
                n_out = int(input_tnsr.get_shape()[1])
            self.params_cnt += 2 * n_out

            beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=input_tnsr.dtype),
                               name='beta', trainable=True, dtype=input_tnsr.dtype)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=input_tnsr.dtype),
                                name='gamma', trainable=True, dtype=input_tnsr.dtype)

            axes = list(np.arange(0, len(input_tnsr.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(input_tnsr, axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.995)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(phase_train,
                                              mean_var_with_update,
                                              lambda: (ema.average(batch_mean), ema.average(batch_var)))

            normed = tf.nn.batch_normalization(input_tnsr, mean, var, beta, gamma, 1e-3)

        return normed

    def _build_conv_unit(self, input_tnsr, is_training, input_features_cnt, output_features_cnt, name):
        with tf.variable_scope('conv_unit' + name):
            filter_weights = self._weight_variable(
                shape=(1, 3, input_features_cnt, output_features_cnt),
                st_dev=np.sqrt(2 / (3 * input_features_cnt))
            )
            b_conv = self._bias_variable([output_features_cnt])

            h_conv = self._conv1d(input_tnsr=input_tnsr, filter_weights=filter_weights) + b_conv

            h_batch_normed = self._batch_norm(input_tnsr=h_conv, is_training=is_training)

            h_res = tf.nn.relu(h_batch_normed)

        return h_res

    def _build_conv_block(self, input_tnsr, is_training, input_features_cnt, output_features_cnt, block_index,
                          item_index):
        with tf.variable_scope('conv_block_%d_%d' % (block_index, item_index)):
            h1 = self._build_conv_unit(
                input_tnsr=input_tnsr,
                is_training=is_training,
                input_features_cnt=input_features_cnt,
                output_features_cnt=output_features_cnt,
                name='1')
            h2 = self._build_conv_unit(
                input_tnsr=h1,
                is_training=is_training,
                input_features_cnt=output_features_cnt,
                output_features_cnt=output_features_cnt,
                name='2')

        return h2

    @staticmethod
    def _temp_max_pool(input_tnsr, name):
        return tf.nn.max_pool(
            value=input_tnsr,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name=name)

    def _gen_initial_embedding(self):
        # def get_random_point_on_a_sphere():
        #     x = np.random.normal(size=EMBEDDING_SIZE)
        #     return x / np.sum(x ** 2) ** 0.5
        # 
        # after_seed = np.random.randint(1 << 30)
        # 
        # np.random.seed(EMBEDDING_GEN_SEED)
        # np_embeddings = np.vstack([get_random_point_on_a_sphere() for i in range(len(ALPHABET_DICT))])
        # np.random.seed(after_seed)
        # 
        # W_embedding = tf.get_variable(
        #     name='embedding',
        #     initializer=tf.constant(value=np_embeddings, dtype=np.float32))

        W_embedding = tf.Variable(
            tf.random_uniform(
                shape=[len(ALPHABET_DICT), EMBEDDING_SIZE],
                minval=-1,
                maxval=1,
                seed=EMBEDDING_GEN_SEED),
            name='embedding')

        # padding token + unknown token
        W_embedding = tf.concat(
            values=(W_embedding,
                    tf.constant(value=0.0, shape=(1, self.embedding_size)),
                    tf.constant(value=0.0, shape=(1, self.embedding_size))),
            axis=0)

        return W_embedding

    def build(self):
        self.network_input = tf.placeholder(
            shape=(None, TEXT_SIZE),
            name='network_input',
            dtype=tf.int32)

        self.correct_labels = tf.placeholder(
            shape=(None, DATASET_NCLASSES[self.data_set_name]),
            name='correct_labels',
            dtype=tf.int32)

        # Store intermediate steps here
        self.h = []
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Prepare initial embedding
        W_init_embedding = self._gen_initial_embedding()
        self.h.append(
            tf.expand_dims(
                input=tf.nn.embedding_lookup(
                    params=W_init_embedding,
                    ids=self.network_input),
                dim=1))

        # First starting convolution has 64 features
        # All the convolutions below are of height 1 and (temporal) width 3;
        #   depth initially equals embedding size (16 by default) and increases
        #   later as tensor widths decreases.
        with tf.name_scope('start_conv'):
            W_start_conv = self._weight_variable(
                shape=(1, 3, self.embedding_size, 64),
                st_dev=np.sqrt(2 / (3 * self.embedding_size)))
            b_start_conv = self._bias_variable([self.feature_cnts[0]])
            self.h.append(
                self._batch_norm(
                    input_tnsr=self._conv1d(self.h[-1], W_start_conv) + b_start_conv,
                    is_training=self.is_training)
            )

        # Build the main part of the network consisting of alternating convolutional blocks
        #   and max poolings.
        # Number of features used in every conv. block defined in self.feature_cnts.
        for i in range(len(self.feature_cnts)):
            # Number of blocks for each feature count defined in self.conv_block_cnts.
            for j in range(self.conv_block_cnts[i]):
                # Create conv. block. Input amount of features equals output number of
                #   features in the previous block.
                self.h.append(
                    self._build_conv_block(
                        input_tnsr=self.h[-1],
                        is_training=self.is_training,
                        input_features_cnt=self.feature_cnts[i - int(j == 0 and i != 0)],
                        output_features_cnt=self.feature_cnts[i],
                        block_index=i + 1,
                        item_index=j + 1))

            # Every set of conv. blocks of equal feature depth is followed by max-pool
            self.h.append(self._temp_max_pool(
                input_tnsr=self.h[-1],
                name='max_pool_' + str(i)))

        # k-max-pool in temporal dimension precedes fully connected layers
        with tf.variable_scope('k_max_pool'):
            # transpose for top_k
            self.h.append(tf.transpose(
                self.h[-1],
                perm=[0, 1, 3, 2],
                name='transpose'))

            # TODO: sorted?
            self.h.append(
                tf.nn.top_k(
                    input=self.h[-1],
                    k=self.k_max_pool_cnt,
                    sorted=True,
                    name='top_k'))

            # reshape for fcn
            self.h.append(
                tf.reshape(
                    tensor=self.h[-1][0],
                    shape=(-1, self.feature_cnts[-1] * self.k_max_pool_cnt),
                    name='reshape'))

        with tf.variable_scope('fc_1'):
            W_fc1 = self._weight_variable(
                shape=(self.feature_cnts[-1] * self.k_max_pool_cnt, self.hidden_layers_cnt),
                st_dev=np.sqrt(2 / (self.feature_cnts[-1] * self.k_max_pool_cnt)))
            W_fc1_loss = tf.nn.l2_loss(W_fc1)
            b_fc1 = self._bias_variable([self.hidden_layers_cnt])

            self.h.append(
                self._batch_norm(
                    input_tnsr=tf.matmul(self.h[-1], W_fc1) + b_fc1,
                    is_training=self.is_training))

            self.h.append(tf.nn.relu(self.h[-1]))

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        if self.use_dropout:
            self.h.append(
                tf.nn.dropout(
                    x=self.h[-1],
                    keep_prob=self.keep_prob,
                    name='drop_out1'))

        with tf.variable_scope('fc_2'):
            W_fc2 = self._weight_variable(
                shape=(self.hidden_layers_cnt, self.hidden_layers_cnt),
                st_dev=np.sqrt(2 / self.hidden_layers_cnt))
            W_fc2_loss = tf.nn.l2_loss(W_fc2)
            b_fc2 = self._bias_variable([self.hidden_layers_cnt])

            self.h.append(
                self._batch_norm(
                    input_tnsr=tf.matmul(self.h[-1], W_fc2) + b_fc2,
                    is_training=self.is_training))

            self.h.append(tf.nn.relu(self.h[-1]))

        if self.use_dropout:
            self.h.append(
                tf.nn.dropout(
                    x=self.h[-1],
                    keep_prob=self.keep_prob,
                    name='drop_out2'))

        with tf.variable_scope('fc_3'):
            W_fc3 = self._weight_variable(
                shape=(self.hidden_layers_cnt, DATASET_NCLASSES[self.data_set_name]),
                st_dev=np.sqrt(2 / self.hidden_layers_cnt))
            W_fc3_loss = tf.nn.l2_loss(W_fc3)
            b_fc3 = self._bias_variable([DATASET_NCLASSES[self.data_set_name]])

            self.h.append(
                tf.add(tf.matmul(a=self.h[-1], b=W_fc3), b_fc3, name='predictions'))

        self.fc_loss = tf.add(W_fc1_loss + W_fc2_loss, W_fc3_loss, name='fc_loss')

        self.predictions = self.h[-1]

        #
        #   evaluation
        #

        self.cross_entropy = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                labels=self.correct_labels,
                logits=self.predictions),
            name='reduced_cross_entropy')
        self.reg_loss = tf.constant(self.reg_coef) * self.fc_loss
        self.loss = self.cross_entropy + self.reg_loss
        # self.loss = self.cross_entropy

        self.global_step = tf.Variable(
            initial_value=0,
            trainable=False,
            dtype=tf.int32,
            name='global_step')
        with tf.name_scope('adam_optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.learn_rate,
                global_step=self.global_step,
                decay_steps=self.lr_decay_freq * (DATASET_SIZE[self.data_set_name] / self.batch_size),
                decay_rate=self.lr_decay_rate,
                staircase=True,
                name='learning_rate')
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss=self.loss,
                global_step=self.global_step)

        self.are_predictions_correct = tf.cast(
            x=tf.equal(
                tf.argmax(
                    input=self.predictions,
                    axis=1),
                tf.argmax(
                    input=self.correct_labels,
                    axis=1)),
            dtype=tf.float32,
            name='are_predictions_correct')
        self.accuracy = tf.reduce_mean(self.are_predictions_correct, name='accuracy')

    def load(self, sess, model_name, test_epoch):
        saver = tf.train.import_meta_graph('checkpoints/' + model_name + '/model-0.meta')
        saver.restore(sess, 'checkpoints/' + model_name + '/model_best-' + str(test_epoch))
        # saver.restore(sess, 'checkpoints/' + model_name + '/model-' + str(test_epoch))

        graph = tf.get_default_graph()

        self.network_input = graph.get_tensor_by_name('network_input:0')
        self.predictions = graph.get_tensor_by_name('fc_3/predictions:0')
        self.correct_labels = graph.get_tensor_by_name('correct_labels:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.is_training = graph.get_tensor_by_name('is_training:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')

    def load_old(self, sess, model_name, test_epoch):
        saver = tf.train.import_meta_graph('checkpoints/' + model_name + '/model-0.meta')
        saver.restore(sess, 'checkpoints/' + model_name + '/model-' + str(test_epoch))

        graph = tf.get_default_graph()

        # x = graph.get_tensor_by_name('x:0')
        # y_ = graph.get_tensor_by_name('y_:0')
        # keep_prob = graph.get_tensor_by_name('keep_prob:0')
        # is_training = graph.get_tensor_by_name('is_training:0')
        # accuracy_op = graph.get_tensor_by_name('accuracy_1:0')

        self.network_input = graph.get_tensor_by_name('x:0')
        self.correct_labels = graph.get_tensor_by_name('y_:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.is_training = graph.get_tensor_by_name('is_training:0')
        self.accuracy = graph.get_tensor_by_name('accuracy_1:0')

    def set_params(self, params):
        for name, value in params.items():
            assert getattr(self, name) is not None

            setattr(self, name, value)
