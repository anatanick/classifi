import os
import copy
import tensorflow as tf
import logging


class CLSTM:
    def __init__(self, model_config):
        self.config = copy.deepcopy(model_config)
        self.model()

        if not os.path.exists(self.config["save_dir"]):
            os.makedirs(self.config["save_dir"])
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    def initialize(self, session):
        ckpt = tf.train.get_checkpoint_state(self.config["save_dir"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return self

    def model(self):
        self.input_x = tf.placeholder(
            tf.int32,
            shape=(None, None),
            name="sequences"
        )

        self.labels = tf.placeholder(
            tf.float32,
            shape=(None, self.config["num_classes"]),
            name="labels"
        )

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.char_emb = tf.Variable(#随机初始化词嵌入向量
                tf.random_uniform(
                    [self.config["vocab_size"], self.config["emb_dim"]],
                    -1.0, 1.0),
                name="char_emb"
            )
            emb_sequences = tf.nn.embedding_lookup(self.char_emb, self.input_x)
            # -1表示在最后一个维度后面增加一个维度
            self.emb_sequences = tf.expand_dims(emb_sequences, -1)

        self.min_features_length = self.config["max_sequence_length"] - max(self.config["filter_sizes"]) + 1
        cnn_features = []#feature map的高度=序列长度-卷积核高度+1
        for filter_size in self.config["filter_sizes"]:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [
                    filter_size,
                    self.config["emb_dim"],
                    1,
                    self.config["channels"]
                ]
                filter_weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weights")
                conv = tf.nn.conv2d(#从句子序列卷积到特征图
                    self.emb_sequences,
                    filter_weights,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2d"
                )
                # [batch_size,channels,sequence_len - filter_size + 1]
                bias = tf.Variable(
                    tf.constant(0.1, shape=[self.config["channels"]]),
                    name="bias"
                )
                conv_proj = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")#映射卷积得到的feature map
                """没有进行pooling，直接squeeze"""
                sequence_features = tf.squeeze(conv_proj, [2])#squeeze默认删除为1的维度
                # 只取最短的
                cnn_features.append(sequence_features[:, :self.min_features_length, :])
        self.cnn_features = tf.concat(cnn_features, -1)#把各个词向量经过不同filter的分量拼接起来

        cell = tf.contrib.rnn.LSTMCell(
            self.config["lstm_dim"],
            forget_bias=1.0,
            state_is_tuple=True,
        )
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config["keep_prob"])
        lstm_initial_state = cell.zero_state(
            tf.shape(self.input_x)[0],
            dtype=tf.float32
        )

        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(
                cell,
                self.cnn_features,
                initial_state=lstm_initial_state,
            )
            self.lstm_final_state = state

        with tf.name_scope("cnn_features_dropout"):
            self.cnn_features_dropout = tf.nn.dropout(
                self.lstm_final_state.h,
                self.keep_prob
            )

        self.loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.config["lstm_dim"], self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer()#Xavier为权重的初始化程序
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="b")
            self.loss += tf.nn.l2_loss(W)
            self.loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.lstm_final_state.h, W, b, name="logits")
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits,
                labels=self.labels
            )
            self.loss = tf.reduce_mean(cross_entropy) + self.loss * self.config["regularizer"]

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32),
                name="accuracy"
            )

        with tf.name_scope("optimizer"):
            self.learning_rate = tf.train.exponential_decay(
                self.config["learning_rate"],
                self.global_step,
                100,
                0.98,
                staircase=True
            )

            if self.config["optimizer"] == "sgd":
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.config["optimizer"] == "adam":
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.config["optimizer"] == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.config["momentum"])
            elif self.config["optimizer"] == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                raise KeyError("Unsupported optimizer '%s'" % (self.config["optimizer"]))

            # 梯度 clip
            grads_and_vars = [
                [tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                for g, v in self.optimizer.compute_gradients(self.loss)
            ]
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=self.global_step)

        with tf.name_scope("smooth_variables"):
            ema = tf.train.ExponentialMovingAverage(#控制模型的更新参数的速度，使模型更稳健robust
                0.9,
                self.global_step,
                name="smooth_ema"
            )
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([self.accuracy]))
            self.smooth_accuracy = ema.average(self.accuracy)
