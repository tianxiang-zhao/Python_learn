import tensorflow as tf
import numpy as np
import pandas as pd

dataset=tf.keras.preprocessing.text_dataset_from_directory(
    "main_directory",
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
)
print(dataset)
for data,label in dataset:
    print(data)
    print(label)
data=data.numpy()
print(data)


tokenizer=tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    char_level=False,
    oov_token=None,
    document_count=0,

)
texts = ["你好 我好 你好 你好 你好 我们 大家 都 好 吗 吗 吗 吗 s", "分词器 训练 文档 训练 文档 文档 你好 我好"]
tokenizer.fit_on_texts(texts)
fre = tokenizer.word_counts  # 统计词频
print("type(fre):\n",type(fre))
print("fre:\n",fre)

# 查看每个词的词频
for i in fre.items():
    print(i[0], " : ", i[1])
# 对词频进行排序
new_fre = sorted(fre.items(), key = lambda i:i[1], reverse = True)
print("new_fre:\n",new_fre)
# # 根据词频进行了升序的排序（注意，词频越大，value越小，这个value不是词频，而是按顺序排列的数字）
order = tokenizer.word_index
print("type(order):\n",type(order))
print("order:\n",order)