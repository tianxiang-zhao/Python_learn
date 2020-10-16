import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
# 顺序模型不适用于以下情况：
#
# 您的模型有多个输入或多个输出
# 您的任何一层都有多个输入或多个输出
# 您需要进行图层共享
# 您需要非线性拓扑（例如，残余连接，多分支模型）

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(1, name="layer3"),
    ]
)
#等效于以下功能：

# Call model on a test input
x = tf.ones((30, 3))
z=tf.ones((30,1))

print(x)
print(z)
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=['mae',"mse"]
)
history=model.fit(x,z,epochs=10,validation_split=0.2)
print(history)
y = tf.ones((1,3))
print(y)
print(model.predict(y))


#预先指定输入形状
layer=layers.Dense(3)
print(layer.weights)#empty

x = tf.ones((1, 4))
y = layer(x)
print(layer.weights)  # Now it has weights, of shape (4, 3) and (3,)

#使用顺序模型进行特征提取
#例如快速创建一个模型以提取顺序模型中所有中间层的输出：
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
# features = feature_extractor(x)
# print(features)

#这是一个类似的示例，仅从一层中提取要素：
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=1, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
print(features)

#使用顺序模型进行学习转移
#首先，假设您有一个顺序模型，并且要冻结除最后一层之外的所有层。在这种情况下，您只需遍历 model.layers并设置layer.trainable = False除最后一层以外的每一层。像这样：
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# Presumably you would want to first load pre-trained weights.
# model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
x=tf.ones((1,784))
y=model(x)
model.summary()


#另一个常见的蓝图是使用顺序模型来堆叠预先训练的模型和一些新初始化的分类层。像这样：
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# Freeze the base model
base_model.trainable = False

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# Compile & train
model.summary()