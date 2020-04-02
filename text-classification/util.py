import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim   #gensim模块下有word2vec，把词汇转换成词向量
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from tensorflow import keras


def pad_sequence(data_dir, pad_length, num_classes):#把每个文本序列填充到同一长度
    labels = []
    data = []
    for line in open(data_dir, "r", encoding="utf-8").readlines():
        item = line.strip().split("\t")#获得每一行的每个标签和文本序列
        if len(item) == 2:
            labels.append(item[0])#把标签存入labels
            data.append([int(x) for x in item[1].split(" ")])#把文本中的每个词汇作为元素加入列表，后列表作为元素加入data

    txt = [x.strip().split(" ")[:pad_length] for x in data]#扩大data列表，下一句填充data
    x_pad = keras.preprocessing.sequence.pad_sequences(data, pad_length, padding='post')
    y_pad = [keras.utils.to_categorical(int(label) - 1, num_classes) for label in labels]
    return x_pad, y_pad


# 读取padding 之后的 csv 文件
def load_data(data_dir, num_classes):
    labels=[]
    content=[]
    with open(data_dir, 'r') as ftrain:
        for line in ftrain:
            fields = line.strip().split()
            label = [0, 0]
            label[int(fields[0])] = 1#生成标签的one-hot编码
            labels.append(label)
            content.append(fields[1:])

    # y_pad = [keras.utils.to_categorical(int(label) - 1, num_classes) for label in labels]
    y_pad = labels
    x_pad = []
    for line in content:
        x_pad.append([int(w) for w in line])
    return x_pad, y_pad


def getWordEmbedding(words, embeddingSize=128):
    """
    按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    """
    # 从word2Vec中读取词向量
    wordVec = gensim.models.KeyedVectors.load_word2vec_format("pretrain/word2vec/word2Vec.bin", binary=True)
    vocab = []
    wordEmbedding = []
    # wordEmbedding.append(np.zeros(embeddingSize))
    # wordEmbedding.append(np.random.randn(embeddingSize))

    print(type(wordVec.wv))
    for word in words:
        try:
            vector = wordVec.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word + "不存在于词向量中")

    return vocab, np.array(wordEmbedding)#返回词汇表，和词嵌入矩阵


def batch_iter(x, y, batch_size):#批处理
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    num_batches = len(x) // batch_size#批数量

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


def get_time_diff(start_time):
    """ 获取已用时间 """
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))
