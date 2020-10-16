import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers

# Auto MPG 数据集
# 该数据集可以从 UCI机器学习库 中获取.



column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv("dataset.txt""", names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset=raw_dataset.copy()
print(dataset.tail())


for i in range(len(dataset)):
    if i%3==0:
        dataset.loc[i,"Origin"]=3
    if i%2==0:
        dataset.loc[i,"Origin"]=2


print(dataset.tail())
#数据清洗 找出一些未知的值

#为了保证这个初始示例的简单性，删除这些行。
dataset=dataset.dropna()
#"Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码 （one-hot）:
origin=dataset.pop("Origin")
dataset["USA"]=(origin==1)*1.0
dataset['Europe']=(origin==2)*1.0
dataset["Japan"]=(origin==3)*1.0
print(dataset)


# 拆分训练数据集和测试数据集
# 现在需要将数据集拆分为一个训练数据集和一个测试数据集。
#
# 我们最后将使用测试数据集对模型进行评估。

trian_dataset=dataset.sample(frac=0.8,random_state=0)
test_dataset=dataset.drop(trian_dataset.index)


# 数据检查
# 快速查看训练集中几对列的联合分布。
sns.pairplot(trian_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],diag_kind='kde')

# 也可以查看总体的数据统计:
train_stats=trian_dataset.describe()
train_stats.pop("MPG")
train_stats=train_stats.transpose()
print(train_stats)

# 从标签中分离特征
# 将特征值从目标值或者"标签"中分离。 这个标签是你使用训练模型进行预测的值

train_labels=trian_dataset.pop("MPG")
test_labels=test_dataset.pop("MPG")


# 数据规范化
# 再次审视下上面的 train_stats 部分，并注意每个特征的范围有什么不同。
#
# 使用不同的尺度和范围对特征归一化是好的实践。尽管模型可能 在没有特征归一化的情况下收敛，它会使得模型训练更加复杂，并会造成生成的模型依赖输入所使用的单位选择。
#
# 注意：尽管我们仅仅从训练集中有意生成这些统计数据，但是这些统计信息也会用于归一化的测试数据集。我们需要这样做，将测试数据集放入到与已经训练过的模型相同的分布中。

def norm(x):
    return (x-train_stats['mean'])/train_stats["std"]
normed_strain_data=norm(trian_dataset)
normed_test_data=norm(test_dataset)
print(normed_test_data)

# 模型
# 构建模型
# 让我们来构建我们自己的模型。这里，我们将会使用一个“顺序”模型，其中包含两个紧密相连的隐藏层，以及返回单个、连续值得输出层。模型的构建步骤包含于一个名叫 'build_model' 的函数中，稍后我们将会创建第二个模型。 两个密集连接的隐藏层。
def build_model():
    model=keras.Sequential([
        layers.Dense(64,activation='relu',input_shape=[len(trian_dataset.keys())]),
        layers.Dense(64,activation="relu"),
        layers.Dense(1)
    ])
    optimizer=tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=
                  optimizer,
                  metrics=['mae',"mse"])
    return model

model=build_model()

# 检查模型
# 使用 .summary 方法来打印该模型的简单描述。

model.summary()

# 现在试用下这个模型。从训练数据中批量获取‘10’条例子并对这些例子调用 model.predict 。e
example_batch=normed_strain_data[:10]
example_result=model.predict(example_batch)
print(example_result)


# 训练模型
# 对模型进行1000个周期的训练，并在 history 对象中记录训练和验证的准确性。
# 通过为每个完成的时期打印一个点来显示训练进度

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch%100==0:print(' ')
        print(".",end=" ")
EPOCHES=300
history=model.fit(
    normed_strain_data,train_labels,
    epochs=EPOCHES,validation_split=0.2,verbose=0,
    callbacks=[PrintDot()]
)
hist=pd.DataFrame(history.history)
hist["epoch"]=history.epoch
print("查看结果")
print(history.history)


def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist["epoch"]=history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("Mean abs error [MPG]")
    plt.plot(hist["epoch"],hist['mae'],
             label="Trian Error")
    plt.plot(hist["epoch"],hist["val_mae"],
             label="Val Error")
    plt.ylim([0,5])
    plt.legend()


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("Mean square error [MPG^2]")
    plt.plot(hist["epoch"], hist['mae'],
             label="Trian Error")
    plt.plot(hist["epoch"], hist["val_mae"],
             label="Val Error")
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)

# 该图表显示在约100个 epochs 之后误差非但没有改进，反而出现恶化。
# 让我们更新 model.fit 调用，当验证值没有提高上是自动停止训练。
# 我们将使用一个 EarlyStopping callback 来测试每个 epoch 的训练条件
# 。如果经过一定数量的 epochs 后没有改进，则自动停止训练。
#
# 你可以从这里学习到更多的回调
#
# 。

# patience 值用来检查改进 epochs 的数量
model=build_model()
early_stop=keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)
history=model.fit(normed_strain_data,train_labels,epochs=EPOCHES,
                  validation_split=0.2,verbose=0,callbacks=[early_stop,PrintDot()])
plot_history(history)

# 如图所示，验证集中的平均的误差通常在 +/- 2 MPG左右。 这个结果好么？ 我们将决定权留给你。
#
# 让我们看看通过使用 测试集 来泛化模型的效果如何，我们在训练模型时没有使用测试集。这告诉我们，当我们在现实世界中使用这个模型时，我们可以期望它预测得有多好。

loss,mae,mse=model.evaluate(normed_test_data,test_labels,verbose=2)
print("Testing set Mean Abs Error :{:5.2f}MPG".format(mae))
#
# 做预测
# 最后，使用测试集中的数据预测 MPG 值:
test_prediction=model.predict(normed_test_data).flatten()
plt.scatter(test_labels,test_prediction)
plt.xlabel=("True values")
plt.ylabel("prediction vlalues")
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100,100],[-100,100])
plt.legend()
plt.show()

# 这看起来我们的模型预测得相当好。我们来看下误差分布。
plt.clf()
error=test_prediction-test_labels
plt.hist(error,bins=25)
plt.ylabel("Count")
plt.legend()
plt.show()













