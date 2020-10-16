import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

dept_targets = np.random.randint(2, size=(6, 4))
print(dept_targets)


dataset=keras.preprocessing.image_dataset_from_directory(
    'mian_directory',batch_size=64,image_size=(2000,2000)
)
for data,labels in dataset:
    print(data.shape)
    print(data.dtype)
    print(labels.shape)
    print(labels.dtype)
#规范化
trianing_data=np.random.randint(0,256,size=(64,200,200,3)).astype("float32")
normalizetion=Normalization(axis=-1)#沿着最后一个下标变换的方向
normalizetion.adapt(trianing_data)
normalizetion_data=normalizetion(trianing_data)
print("var:%.4f" %np.var(normalizetion_data))#np.var求标准差
print("mean:%.4f" %np.mean(normalizetion_data))
#重新缩放和中心裁剪图像
cropper=CenterCrop(height=150,width=150)
scaler=Rescaling(scale=1.0/255)
output_data=scaler(cropper(trianing_data))#无论是Rescaling层与CenterCrop层是无状态的，所以没有必要调用adapt()在这种情况下。
print(output_data.shape)
print("min:",np.min(output_data))


#使用Keras Functional API构建模型
dense=keras.layers.Dense(units=16)#它将其输入映射到16维特征空间：

#但是用于任何大小的RGB图像的输入将具有shape (None, None, 3)。
inputs=keras.Input(shape=(None,None,3))
from tensorflow.keras import layers
x=CenterCrop(height=150,width=150)(inputs)
x=Rescaling(scale=1.0/255)(x)
x=layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu")(x)
x=layers.MaxPooling2D(pool_size=(3,3))(x)
x=layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu")(x)
x=layers.MaxPooling2D(pool_size=(3,3))(x)
x=layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu")(x)
x=layers.MaxPooling2D(pool_size=(3,3))(x)

x=layers.GlobalAveragePooling2D()(x)

num_classes=10
outputs=layers.Dense(num_classes,activation="softmax")(x)
model=keras.Model(inputs=inputs,outputs=outputs)
data=np.random.randint(0,256,size=(64,200,200,3)).astype("float32")
processed_data=model(data)
model.summary()
print(model.summary())
print(processed_data)

#训练模型 fit()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              loss=keras.losses.CategoricalCrossentropy())


