import os
import matplotlib.pyplot as plt

import tensorflow as tf

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
# 创建一个 tf.data.Dataset
# TensorFlow的 Dataset API 可处理在向模型加载数据时遇到的许多常见情况。这是一种高阶 API ，用于读取数据并将其转换为可供训练使用的格式。
# 如需了解详情，请参阅数据集快速入门指南
#
# 由于数据集是 CSV 格式的文本文件，请使用 make_csv_dataset 函数将数据解析为合适的格式。由于此函数为训练模型生成数据，默认行为是对数据进行随机处理
# （shuffle=True, shuffle_buffer_size=10000），并且无限期重复数据集（num_epochs=None）。 我们还设置了 batch_size 参数:
column_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
feature_names=column_names[:-1]
label_name=column_names[-1]
batch_size=32
train_data=tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

# make_csv_dataset 返回一个(features, label) 对构建的 tf.data.Dataset ，其中 features 是一个字典: {'feature_name': value}
#
# 这些 Dataset 对象是可迭代的。 我们来看看下面的一些特征:

features,labels=next(iter(train_data))
print(features)
print(labels)
# 注意到具有相似特征的样本会归为一组，即分为一批。更改 batch_size 可以设置存储在这些特征数组中的样本数。
#
# 绘制该批次中的几个特征后，就会开始看到一些集群现象：
plt.scatter(
    features['petal_length'],
    features['sepal_length'],
    c=labels,
    cmap='viridis'
)
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.show()
# 要简化模型构建步骤，请创建一个函数以将特征字典重新打包为形状为 (batch_size, num_features) 的单个数组。
#
# 此函数使用 tf.stack 方法，该方法从张量列表中获取值，并创建指定维度的组合张量:

def pack_features_vector(features,lables):
    features=tf.stack(list(features.values()),axis=1)
    return features,lables
# 然后使用 tf.data.Dataset.map 方法将每个 (features,label) 对中的 features 打包到训练数据集中：
train_dataset=train_data.map(pack_features_vector)
features,labels=next(iter(train_dataset))
print(features[:5])
print(features.shape)

# tf.keras.Sequential 模型是层的线性堆叠。该模型的构造函数会采用一系列层实例；
# 在本示例中，采用的是 2 个密集层（各自包含10个节点）,以及 1 个输出层（包含 3 个代表标签预测的节点。
# 第一个层的 input_shape 参数对应该数据集中的特征数量，它是一项必需参数：
model=tf.keras.Sequential([

    tf.keras.layers.Dense(10,activation=tf.nn.relu,input_shape=[4]),
    tf.keras.layers.Dense(10,activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])
predictions=model(features)
print(predictions[:5])
# 在此示例中，每个样本针对每个类别返回一个 logit。
#
# 要将这些对数转换为每个类别的概率，请使用 softmax 函数:

print(tf.nn.softmax(predictions[:5]))

# 对每个类别执行 tf.argmax 运算可得出预测的类别索引。不过，该模型尚未接受训练，因此这些预测并不理想。
print("prediction:{}".format(tf.argmax(predictions,axis=1)))
print("lables:{}".format(labels))
#鸢尾花分类问题是监督式机器学习的一个示例: 模型通过包含标签的样本加以训练。 而在非监督式机器学习中，样本不包含标签。相反，模型通常会在特征中发现一些规律。


# 定义损失和梯度函数
# 在训练和评估阶段，我们都需要计算模型的损失。 这样可以衡量模型的预测结果与预期标签有多大偏差，也就是说，模型的效果有多差。我们希望尽可能减小或优化这个值。

# 我们的模型会使用 tf.keras.losses.SparseCategoricalCrossentropy 函数计算其损失，此函数会接受模型的类别概率预测结果和预期标签，然后返回样本的平均损失。

loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model,x,y):
    y_=model(x)
    return loss_object(y_true=y,y_pred=y_)
l=loss(model,features,labels)
print("Loss test:{}".format(l))

# 使用 tf.GradientTape 的前后关系来计算梯度以优化你的模型:

def grad(model,input,targets):
    with tf.GradientTape() as tape:
        loss_value=loss(model,input,targets)
    return loss_value,tape.gradient(loss_value,model.trainable_variables)

# TensorFlow有许多可用于训练的优化算法。此模型使用的是 tf.train.GradientDescentOptimizer ， 它可以实现随机梯度下降法（SGD）。
# learning_rate 被用于设置每次迭代（向下行走）的步长。 这是一个 超参数 ，您通常需要调整此参数以获得更好的结果。
#
# 我们来设置优化器
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
#我们将使用它来计算单个优化步骤
loss_value,grads=grad(model,features,labels)
print("step:{}， Initial Loss:{}".format(optimizer.iterations.numpy(),loss_value.numpy()))
optimizer.apply_gradients(zip(grads,model.trainable_variables))
print("step:{}， Initial Loss:{}".format(optimizer.iterations.numpy(),loss(model,features,labels).numpy()))

# 训练循环
# 一切准备就绪后，就可以开始训练模型了！训练循环会将数据集样本馈送到模型中，以帮助模型做出更好的预测。以下代码块可设置这些训练步骤：
#
# 迭代每个周期。通过一次数据集即为一个周期。
# 在一个周期中，遍历训练 Dataset 中的每个样本，并获取样本的特征（x）和标签（y）。
# 根据样本的特征进行预测，并比较预测结果和标签。衡量预测结果的不准确性，并使用所得的值计算模型的损失和梯度。
# 使用 optimizer 更新模型的变量。
# 跟踪一些统计信息以进行可视化。
# 对每个周期重复执行以上步骤。
# num_epochs 变量是遍历数据集集合的次数。与直觉恰恰相反的是，训练模型的时间越长，并不能保证模型就越好。num_epochs 是一个可以调整的超参数。选择正确的次数通常需要一定的经验和实验基础。

# 保留结果用于绘制
trian_loss_results=[]
train_accuracy_results=[]
num_epochs=201
for epoch in range(num_epochs):
    epoch_loss_avg=tf.keras.metrics.Mean()
    epoch_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

    for x,y in train_dataset:
    #优化模型
        loss_value,grads=grad(model,x,y)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
    #追踪进度
        epoch_loss_avg(loss_value)#添加当前的batch loss
    # 比较预测标签与真实标签
        epoch_accuracy(y,model(x))
    #循环结束
    trian_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch%50==0:
        print("Epoch {:03d}: Loss:{:0.3f},Accuracy:{:0.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))


# 可视化损失函数随时间推移而变化的情况
# 虽然输出模型的训练过程有帮助，但查看这一过程往往更有帮助。 TensorBoard 是与 TensorFlow 封装在一起的出色可视化工具，不过我们可以使用 matplotlib 模块创建基本图表。
#
# 解读这些图表需要一定的经验，不过您确实希望看到损失下降且准确率上升。
fig,axes=plt.subplots(2,sharex=True,figsize=(12,8))
fig.suptitle("Trian Metrics")

axes[0].set_ylabel("Loss",fontsize=14)
axes[0].plot(trian_loss_results)

axes[1].set_ylabel("Accuracy",fontsize=14)
axes[1].plot(train_accuracy_results)
axes[1].set_xlabel("Epoch",fontsize=14)
plt.show()

# 建立测试数据集
# 评估模型与训练模型相似。最大的区别在于，样本来自一个单独的测试集，而不是训练集。为了公正地评估模型的效果，用于评估模型的样本务必与用于训练模型的样本不同。
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset=tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False

)

test_dataset=test_dataset.map(pack_features_vector)

# 根据测试数据集评估模型
# 与训练阶段不同，模型仅评估测试数据的一个周期。在以下代码单元格中，我们会遍历测试集中的每个样本，然后将模型的预测结果与实际标签进行比较。这是为了衡量模型在整个测试集中的准确率。

test_accuracy=tf.keras.metrics.Accuracy()
for x,y in test_dataset:
    logits=model(x)
    prediction=tf.argmax(logits,axis=1,output_type=tf.int32)
    test_accuracy(prediction,y)
print("Test set accuracy:{:.3%}".format(test_accuracy.result()))

print(tf.stack([y,prediction]))
# 使用经过训练的模型进行预测
# 我们已经训练了一个模型并“证明”它是有效的，但在对鸢尾花品种进行分类方面，这还不够。现在，我们使用经过训练的模型对 无标签样本（即包含特征但不包含标签的样本）进行一些预测。
#
# 在现实生活中，无标签样本可能来自很多不同的来源，包括应用、CSV 文件和数据 Feed。暂时我们将手动提供三个无标签样本以预测其标签。回想一下，标签编号会映射到一个指定的表示法：
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])
predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  print(logits)
  class_idx = tf.argmax(logits).numpy()
  print(class_idx)
  p = tf.nn.softmax(logits)[class_idx]#预测的概率值，通过上面获取的最大概率值得下标获取
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
