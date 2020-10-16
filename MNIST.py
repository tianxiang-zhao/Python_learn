import tensorflow as tf
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
print(len(train_x))
print(len(test_x))
#输出图像数据和标记数据形状
print("train_x",train_x.shape,train_x.dtype)#六万条数据 每个数据为28*28的数组
print(train_x[0])
import matplotlib.pylab as plt

# plt.axis("off")
# plt.imshow(train_x[0],cmap="gray")
# plt.show()
import numpy as np
for i in range(4):
    num=np.random.randint(1,60000)
    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(train_x[num],cmap="pink")
    plt.title(train_y[num])

plt.show()