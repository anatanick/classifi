import logging
import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
from model.CLSTM import CLSTM
from model.RCNN import  RCNN
from model.TextCNN import TextCNN
from model.BiLSTM_Attention import BiLSTM_Attention
import time
from datetime import timedelta
from util import batch_iter

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def load_data(data_path):
    logging.basicConfig(level=logging.INFO)

    max_sequence_length = None
    data_dict = dict()
    for data_type in ["train", "dev"]:
        label_list = list()
        sequence_list = list()

        with open(os.path.join(data_path, data_type), 'r') as ftrain:
            for line in ftrain:
                fields = line.strip().split()
                label = [0, 0]
                label[int(fields[0])] = 1#把标签换成one-hot编码
                label_list.append(label)
                sequence_list.append(fields[1:])

        padded_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences(
            sequence_list,
            padding="post",
            value=0,
            maxlen=max_sequence_length,
        )

        if max_sequence_length is None:
            max_sequence_length = padded_sequences.shape[1]

        data_dict[data_type] = [np.asarray(label_list), padded_sequences]
    return data_dict,int(max_sequence_length)


def train(data_dict):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = model_config["tensorboard_dir"]
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session = tf.Session(config=session_conf)

        with session.as_default():
            if model_name == "clstm":
                model = CLSTM(model_config)
            elif model_name == "textcnn":
                model = TextCNN(model_config)
            elif model_name == 'rcnn':
                model = RCNN(model_config)
            elif model_name == 'bilstm_attention':
                model = BiLSTM_Attention(model_config)
            model.initialize(session)

            tf.summary.scalar("loss", model.loss)
            tf.summary.scalar("accuracy", model.accuracy)
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tensorboard_dir)
            logging.info("Training ...")

            train_labels, train_sequences = data_dict["train"]
            dev_labels, dev_sequences = data_dict["dev"]

            best_acc_val = 0.0  # 最佳验证集准确率
            last_improved = 0  # 记录上一次提升批次
            start_time = time.time()
            flag = False
            batch_size = model.config["batch_size"]

            for epoch in range(model.config["epoches"]):
                print("Epoch" + str(epoch + 1))
                for batch_sequence,batch_labels in batch_iter(train_sequences, train_labels, batch_size):

                    feed_dict = {
                        model.input_x: batch_sequence,
                        model.labels: batch_labels,
                        model.keep_prob: model.config["keep_prob"]
                    }

                    train_acc, _ = session.run([model.accuracy, model.train_op], feed_dict=feed_dict)
                    global_step = session.run(model.global_step)

                    if global_step % model.config["print_every"] == 0:
                        # train_acc = session.run(model.smooth_accuracy)
                        dev_acc = evaluate(model, session, dev_sequences, dev_labels)

                        # 每多少轮次将训练结果写入tensorboard scalar
                        s = session.run(merged_summary, feed_dict=feed_dict)
                        writer.add_summary(s, global_step)

                        if dev_acc > best_acc_val:
                            best_acc_val, last_improved = dev_acc, global_step
                            improved_str = "*"#如果验证集准确率大于最佳准确率，则更新最佳准确率，并用*标记

                            if not os.path.exists(model.config["save_dir"]):
                                os.makedirs(model.config["save_dir"])
                            model.saver.save(
                                session,
                                os.path.join(model.config["save_dir"], "textcnn.ckpt"),
                                global_step=global_step,
                            )
                        else:
                            improved_str = ""
                        time_dif = get_time_dif(start_time)
                        logs = "step {},Train acc {:>6.2%}, Dev acc {:>6.2%}, Time {}  {}"
                        print(logs.format(global_step, train_acc,dev_acc, time_dif, improved_str))
                        start_time = time.time()
                    # early-stopping
                    if global_step - last_improved > model.config["require_improvement"]:
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break
                if flag:
                    break


def predict(test_dir, max_sequence_length, save_dir):
    logging.info("Predicting ...")

    if model_name == "clstm":
        model = CLSTM(model_config)
    elif model_name == "textcnn":
        model = TextCNN(model_config)
    elif model_name == 'rcnn':
        model = RCNN(model_config)
    elif model_name == 'bilstm_attention':
        model = BiLSTM_Attention(model_config)
    label_list = []
    sequence_list = []

    with open(test_dir, 'r') as ftrain:
        for line in ftrain:
            fields = line.strip().split()
            label = [0, 0]
            label[int(fields[0])] = 1#把标签换为one-hot编码
            label_list.append(label)
            sequence_list.append(fields[1:])

    padded_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences(
        sequence_list,
        padding="post",
        value=0,
        maxlen=max_sequence_length,
    )

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session = tf.Session(config=session_conf)
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_dir)  # 读取保存的模型

    test_acc = evaluate(model, session, padded_sequences, label_list)

    msg = 'Test Acc: {:>6.2%}'
    print(msg.format(test_acc))

    batch_size = model_config["batch_size"]
    data_len = len(padded_sequences)
    num_batch = int((data_len - 1) / batch_size) + 1
    labels = np.argmax(label_list, 1)
    prediction = np.zeros(shape=len(padded_sequences), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: padded_sequences[start_id:end_id],
            model.keep_prob: 1.0#预测时，神经元全保留
        }
        prediction[start_id:end_id] = session.run(model.prediction, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, prediction, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(labels, prediction)
    print(cm)


def evaluate(model, session, sequences, labels):
    correct_samples = 0
    batch_size = model.config["batch_size"]
    end = len(sequences)
    for batch_sequence, batch_labels in batch_iter(sequences, labels, batch_size):
        batch_length = len(batch_sequence)
        feed_dict = {
            model.input_x: batch_sequence,
            model.labels: batch_labels,
            model.keep_prob: 1.0,
        }

        batch_acc = session.run(model.accuracy, feed_dict=feed_dict)
        correct_samples += batch_length * batch_acc
    return correct_samples / sequences.shape[0]


if __name__ == '__main__':
    categories = ["0", "1"]
    model_name = "bilstm_attention"

    print("model:"+model_name)
    data_path = "data/messages"
    data_dict,max_sequence_length = load_data(data_path)
    model_config = {
        "vocab_size": 5230,
        "max_sequence_length": max_sequence_length,
        "emb_dim": 200,
        "channels": 64,
        "filter_sizes": [3, 4, 5],
        "keep_prob": 0.8,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "epoches": 3,
        "num_classes": 2,
        "require_improvement": 2000,
        "print_every": 20,
        "save_dir": os.path.join("checkpoints" , model_name),
        "tensorboard_dir": os.path.join("tensorboard" , model_name),
        "clip": 5.0,
        "optimizer": "adam",
        "lstm_dim":64,
        "regularizer": 0.1,
        "momentum":0.9,
        "output_size":128,
    }
    train(data_dict)

    latest_model = tf.train.latest_checkpoint(model_config["save_dir"])
    predict(os.path.join(data_path,"test"), max_sequence_length, latest_model)
#CLSTM | AdamOptimizer | Dev acc 在第2个Epoch 开始增长
