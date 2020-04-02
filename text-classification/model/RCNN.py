import os
import copy
import tensorflow as tf
import logging


class RCNN:
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
            self.char_emb = tf.Variable(
                tf.random_uniform(#随机初始化词嵌入向量
                    [self.config["vocab_size"], self.config["emb_dim"]],
                    -1.0, 1.0),
                name="char_emb"
            )
            self.embedded_words = tf.nn.embedding_lookup(self.char_emb, self.input_x)
            # 复制一份
            self.embedded_words_ = self.embedded_words

        with tf.name_scope("Bi-LSTM"):#构建双向LSTM
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.name_scope(direction):
                    lstm_cell[direction] = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(
                            self.config["lstm_dim"],
                            use_peepholes=True,
                            initializer=tf.contrib.layers.xavier_initializer(),#初始化权重
                            state_is_tuple=True
                        ))

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                self.embedded_words,
                dtype=tf.float32
            )
            # 对outputs中的forward和backward的结果拼接，传入到下一层Bi-LSTM中
            # [batch_size, sequence_length, lstm_dim * 2]
            self.embedded_words = tf.concat(outputs, axis=2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        forward_out, backward_out = tf.split(self.embedded_words, 2, -1)

        with tf.name_scope("context"):
            shape = [tf.shape(forward_out)[0], 1, tf.shape(forward_out)[2]]
            self.cleft = tf.concat([tf.zeros(shape), forward_out[:, :-1]], axis=1, name="contextLeft")
            self.cright = tf.concat([backward_out[:, 1:], tf.zeros(shape)], axis=1, name="contextRight")

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("word_rep"):
            self.word_rep = tf.concat([self.cleft, self.embedded_words_, self.cright], axis=2)
            word_length = self.config["lstm_dim"] * 2 +  self.config["emb_dim"]#每一个词向量表示的长度

        with tf.name_scope("text_rep"):
            output_size = self.config["output_size"]
            textW = tf.Variable(tf.random_uniform([word_length, output_size], -1.0, 1.0), name="W2")
            textB = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b2")

            # tf.einsum可以指定维度的消除运算
            self.text_rep = tf.tanh(tf.einsum('aij,jk->aik', self.word_rep, textW) + textB)

        # 做max-pooling 操作，将时间步的维度消失
        output = tf.reduce_max(self.text_rep, axis=1)

        self.loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer()#初始化权重
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="b")
            self.loss += tf.nn.l2_loss(W)
            self.loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(output, W, b, name="logits")
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
            ema = tf.train.ExponentialMovingAverage(
                0.9,
                self.global_step,
                name="smooth_ema"
            )
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([self.accuracy]))
            self.smooth_accuracy = ema.average(self.accuracy)
