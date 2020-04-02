import pandas as pd
import re
import jieba
import os
from tqdm import *
from collections import Counter
from sklearn.utils import shuffle

def load_stop(stop_dir):#加载停止词
	stop_list = []
	for line in open(stop_dir, "r", encoding="utf-8").readlines():
		stop_list.append(line.strip())#去掉这一行前后空格，并添加到stop_list中
	return stop_list


def clean_txt(txt):
	txt = str.lower(txt)#把文本中的词都换成小写，下一句是匹配url
	txt = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","URL",txt)
	pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")#匹配数字、字母，要和findall、match搭配使用
	return pattern.sub("", txt.strip())


def load_vocab(vocab_dir, train_dir, stop_dir, vocabulary_size=10000, min_freq=5, stop=0, delimiter="\t"):
	print("Loading vocab...")#加载模型词典，输入词典目录，训练集目录，停止词目录，词典大小10000，最小频数5，分割符tab
	if not os.path.exists(vocab_dir):
		print("Building vocab...")
		with open(train_dir, "r", encoding="utf-8") as f:
			words = []
			# 是否去停用词
			if stop:
				stop_words = load_stop(stop_dir)
				for item in tqdm(f.readlines()):
					line_content = item.strip().split(delimiter)
					if len(line_content)!= 2:#如果不是“序列	文本”格式的行，直接跳过
						continue
					#如果词不在停用词文本中，且不是数字，就把词放入words列表
					words += [w for w in jieba.cut(clean_txt(line_content[1])) if w not in stop_words and not w.isdigit()]
			else:#不去停用词
				for item in tqdm(f.readlines()):
					line_content = item.strip().split(delimiter)
					if len(line_content)!= 2:
						continue
					#如果词不是数字，就把词放入words列表
					words += [x for x in jieba.cut(clean_txt(line_content[1])) if not x.isdigit()]
			counter = Counter(words)#统计所有词的个数
			vocab = ["PAD", "UNK","NUM"] + [w for _, (w, count) in enumerate(counter.most_common(vocabulary_size - 3)) if count > min_freq]
		with open(vocab_dir, "w", encoding="utf-8") as f:#上句，按照频数给所有词排序，空出前三个元素，给填充，未知，数字
			f.write("\n".join(vocab))	#将vocab列表中的元素按换行分隔，写入vocab.txt
	else:#如果已经存在vocab.txt，直接读取到vocab列表
		with open(vocab_dir, "r", encoding="utf-8") as f:
			vocab = [w.strip() for w in f.readlines()]
	return dict(zip(vocab, range(len(vocab))))#返回词汇表词典，key是词汇，value是该词汇的序号


def txt2csv(data_dir, output_dir, stop=0):#把文本转成csv格式，即转化成数字
	txt = []
	vocab = load_vocab(vocab_dir, train_dir, stop_dir, stop=stop)#加载词典

	print("Transferring txt2id...")
	for line in tqdm(open(data_dir, "r", encoding="utf-8").readlines()):
		items = line.strip().split("\t")
		if len(items) != 2:
			continue
		text = clean_txt(items[1])#获得文本内容
		if text == "":
			continue

		if stop:#去除停用词
			stop_list = load_stop(stop_dir)
			content = [x for x in jieba.cut(text) if x not in stop_list]#分词且去除停用词
		else:
			content = [x for x in jieba.cut(text)]#只分词
		items[0] = label_dict[items[0]]#把标签对应的value值保存在items[0]
		items[1] = " ".join([str(vocab["NUM"]) if str(w).isdigit() else str(vocab.get(str(w), vocab['UNK'])) for w in content ])
		txt.append(items)#上句就把词汇转成了词汇表中的数字
	train_pivot = int(len(txt)*0.7)
	dev_pivot = int(len(txt)*0.9)

	data_txt = shuffle(txt)
	train_txt = data_txt[:train_pivot]
	dev_txt = data_txt[train_pivot:dev_pivot]
	test_txt = data_txt[dev_pivot:]
	with open("data/messages/train","w",encoding="utf-8") as f:
		for line in train_txt:
			f.write("\t".join(line)+"\n")
	with open("data/messages/dev","w",encoding="utf-8") as f:
		for line in dev_txt:
			f.write("\t".join(line)+"\n")
	with open("data/messages/test","w",encoding="utf-8") as f:
		for line in test_txt:
			f.write("\t".join(line)+"\n")
	print("finished")
	# return pd.DataFrame(txt, columns=["label", "content"], index=None).to_csv(output_dir, index=None)



if __name__ == "__main__":
	"""messages"""
	stop_dir = "data/stop.txt"
	vocab_dir = "data/messages/vocab.txt"
	train_dir = "data/messages/train.txt"
	test_dir = "data/messages/test.txt"
	label_dict={
		"0":"0",
		"1":"1",
	}
	jieba.load_userdict("data/user_dict.txt")
	txt2csv(train_dir,"data/message/")


	"""cnews"""



	# label_dict = {"体育": "1",
	# 			  "财经": "2",
	# 			  "房产": "3",
	# 			  "家居": "4",
	# 			  "教育": "5",
	# 			  "科技": "6",
	# 			  "时尚": "7",
	# 			  "时政": "8",
	# 			  "游戏": "9",
	# 			  "娱乐": "10",
	# 			  }

	# for type in ["train", "val", "test"]:
	# 	print("【" + type + " data】")
	# 	txt2csv("data/cnews/cnews." + type + ".txt", "data/" + type + ".csv", stop=1)
	# print("txt2csv finished...")
