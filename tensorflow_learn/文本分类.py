import tensorflow as tf
from  tensorflow import  keras
import  numpy as np
print(tf.__version__)
# 下载 IMDB 数据集
imdb=keras.datasets.imdb
(train_data,train_label),(test_data,test_label)=imdb.load_data(num_words=10000)

print(len(train_data))
# 将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。
# 例如，序列[3, 5]将转换为一个 10,000 维的向量，该向量除了索引为 3 和 5 的位置是 1 以外，其他都为 0。
# 然后，将其作为网络的首层——一个可以处理浮点型向量数据的稠密层。
# 不过，这种方法需要大量的内存，需要一个大小为 num_words * num_reviews 的矩阵。

# 或者，我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。
# 我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()
word_index["<PDA>"]=0

train_data=keras.preprocessing.sequence.pad_sequences(train_data,
                                                      value=word_index["<PDA>"],
                                                      padding="post",
                                                      maxlen=256
                                                     )
test_data=keras.preprocessing.sequence.pad_sequences(test_data,
                                                      value=word_index["<PDA>"],
                                                      padding="post",
                                                      maxlen=256
                                                     )
print(len(test_data[0]),test_data[1])

# 构建模型
# 神经网络由堆叠的层来构建，这需要从两个主要方面来进行体系结构决策：
#
# 模型里有多少层？
# 每个层里有多少隐层单元（hidden units）？
# 在此样本中，输入数据包含一个单词索引的数组。要预测的标签为 0 或 1。让我们来为该问题构建一个模型：

vocab_size=10000
model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,30))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(30,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

# 第一层是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习到的。
# 向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding)。
# 接下来，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
# 该定长输出向量通过一个有 30个隐层单元的全连接（Dense）层传输。
# 最后一层与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。

# 隐层单元
# 上述模型在输入输出之间有两个中间层或“隐藏层”。输出（单元，结点或神经元）的数量即为层表示空间的维度。换句话说，是学习内部表示时网络所允许的自由度。
#
# 如果模型具有更多的隐层单元（更高维度的表示空间）和/或更多层，则可以学习到更复杂的表示。
# 但是，这会使网络的计算成本更高，并且可能导致学习到不需要的模式——一些能够在训练数据上而不是测试数据上改善性能的模式。
# 这被称为过拟合（overfitting），我们稍后会对此进行探究。

# 损失函数与优化器
# 一个模型需要损失函数和优化器来进行训练。由于这是一个二分类问题且模型输出概率值（一个使用 sigmoid 激活函数的单一单元层），我们将使用 binary_crossentropy 损失函数。
#
# 这不是损失函数的唯一选择，例如，您可以选择 mean_squared_error 。但是，一般来说 binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”，或者在我们的示例中，指的是度量 ground-truth 分布与预测值之间的“距离”。
#
# 稍后，当我们研究回归问题（例如，预测房价）时，我们将介绍如何使用另一种叫做均方误差的损失函数。
#
# 现在，配置模型来使用优化器和损失函数：
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

# 创建一个验证集
# 在训练时，我们想要检查模型在未见过的数据上的准确率（accuracy）。
# 通过从原始训练数据中分离 10,000 个样本来创建一个验证集。
# （为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））。


x_val=train_data[:10000]
partial_x_train=train_data[10000:]


y_val=train_label[:10000]
partial_y_train=train_label[10000:]


history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=30,
                  batch_size=512,
                  validation_data=(x_val,y_val),
                  verbose=1

)
result=model.evaluate(test_data,test_label,verbose=2)
print(result)


# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件：
history_dict=history.history
print(history_dict.keys())
import  matplotlib.pyplot as plt
acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict["val_loss"]
epoches=range(1,len(acc)+1)
#bo 代表蓝点
plt.plot(epoches,loss,'bo',label="Train_loss")
#b代表蓝线
plt.plot(epoches,val_loss,'b',label="validation loss")
plt.title("Trian ad validation loss")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()
plt.show()

#清空数据
plt.clf()

plt.plot(epoches,acc,'bo',label="Trian acc")
plt.plot(epoches,val_acc,'b',label="Validation acc")
plt.title("Train add validation accuracy")
plt.xlabel('Epoches')
plt.ylabel("Accuracy")
plt.legend()
plt.show()