import tensorflow as tf
from  tensorflow import keras
import numpy as  np
import matplotlib.pylab as plt
print(tf.__version__)

#导入fashion mnist数据集
fashion_mnist=keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

class_name=["T衫","裤子","套头衣","连衣裙","外套","凉鞋","衬衫","运动鞋","包","短靴"]
print(x_train.shape)
print(len(y_train))

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

x_train=x_train/255.0
x_test=x_test/255.0

plt.figure(figsize=(10,10))
plt.rcParams['font.sans-serif']="SimHei"#设置默认字体为黑体
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i],cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel(class_name[y_train[i]])
plt.show()


#构建神经网络
#

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),#它们是密集连接或全连接神经层。第一个 Dense 层有 128 个节点（或神经元）
    keras.layers.Dense(10)#返回一个长度为 10 的 logits 数组。每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。

])
# 编译模型
# 在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：
#
# 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。

model.compile(optimizer='adam',#还需一个一个了解，参数的含义
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#训练模型
# 训练神经网络模型需要执行以下步骤：
#
# 将训练数据馈送给模型。在本例中，训练数据位于 train_images 和 train_labels 数组中。
# 模型学习将图像和标签关联起来。
# 要求模型对测试集（在本例中为 test_images 数组）进行预测。
# 验证预测是否与 test_labels 数组中的标签相匹配。
# 向模型馈送数据
# 要开始训练，请调用 model.fit 方法，这样命名是因为该方法会将模型与训练数据进行“拟合”：
model.load_weights("img_model.hdf5")#直接加载训练好的模型
# model.fit(x_train,y_train,epochs=5)
#评估准确率
# 接下来，比较模型在测试数据集上的表现：
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print("\nTest accuracy：",test_acc)

# 进行预测
# 在模型经过训练后，您可以使用它对一些图像进行预测。模型具有线性输出，即 logits。您可以附加一个 softmax 层，将 logits 转换成更容易理解的概率。
probability_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions=probability_model.predict(x_test)
model.save_weights('img_model.hdf5')

print(predictions[0])
# 预测结果是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”。您可以看到哪个标签的置信度值最大：
print(np.argmax(predictions[0]))

# 您可以将其绘制成图表，看看模型对于全部 10 个类的预测。
def plot_image(i,predictions_array,true_label,img):
    predictions_array,true_label,img=predictions_array,true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label=np.argmax(predictions_array)
    if predicted_label==true_label:
        color="blue"
    else:
        color="red"
    plt.xlabel("{}{:2.0f}%({})".format(class_name[predicted_label],
                                      100*np.max(predictions_array),
                                      class_name[true_label]),

                                      color=color)
def plot_value_array(i,predictons_array,true_label):
    predictons_array,true_label=predictons_array,true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot=plt.bar(range(10),predictons_array,color="#777777")
    plt.ylim([0,1])
    predicted_label=np.argmax(predictons_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i=0
plt.figure(figsize=(6,3))
plt.subplot(121)
plot_image(i,predictions[i],y_test,x_train)
plt.subplot(122)
plot_value_array(i,predictions[i],y_test)
plt.show()

num_rows=5
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],y_test,x_test)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],y_test)
plt.tight_layout()
plt.show()

#使用训练好的模型
# 最后，使用训练好的模型对单个图像进行预测。
img=x_test[1]
print(img.shape)
img=(np.expand_dims(img,0))
print(img.shape)

predictions_single=probability_model.predict(img)
print(predictions_single)
plt.figure(figsize=(6,3))

plot_value_array(1,predictions_single[0],y_test)
plt.xticks(range(10),class_name,rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))

