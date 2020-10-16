import  tensorflow as tf
print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.square(2)+tf.square(4))
#每个tf.Tensor都有一个形状和一个数据类型：
x=tf.matmul([[1]],[[2,3]])
print(x)
print(x.shape)
print(x.dtype)
y=tf.matmul([[1,1]],[[1,2,3],[1,2,3]])
print(y.numpy())

#生成几个三乘三的矩阵，用1填充
import numpy as np
ndarray=np.ones([3,3])
print(ndarray)
tensor=tf.multiply(ndarray,42)
print(tensor)
print(tf.add(tensor,1))
print(tensor.numpy())
print(type(tensor))

#GPU加速
# 使用GPU进行计算可加速许多TensorFlow操作。没有任何注释，TensorFlow会自动决定是使用GPU还是CPU进行操作-如有必要，在CPU和GPU内存之间复制张量。
# 由操作产生的张量通常由执行操作的设备的内存支持，例如：
x=tf.random.uniform([3,3])
print("Is there a GPU available: ")
print(tf.config.experimental.list_physical_devices("CPU"))
print(x.device.endswith('CPU:0'))
print(x)

import time
def time_malmul(x):
    start=time.time()
    for loop in range(10):
        tf.matmul(x,x)
    result=time.time()-start
    print("10 loops {:0.2f}ms".format(1000*result))
    #force execution on CPU
print("On Cpu")
with tf.device('CPU:0'):
    x=tf.random.uniform([1000,1000])
    assert  x.device.endswith("CPU:0")
    time_malmul(x)
    print(x)