from keras.datasets import cifar10
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np

train_ds = keras.preprocessing.image_dataset_from_directory(
    directory='../mian_directory/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
# for data,label in train_ds:
#     print(label)
#     print(data)
for data,label in train_ds:
    print(label)
print(train_ds)
validation_ds = keras.preprocessing.image_dataset_from_directory(
    directory='../mian_directory/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256))




model = keras.applications.Xception(weights=None, input_shape=(256, 256, 3),classes=1)#一共两种类型图片
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_ds, epochs=2, validation_data=validation_ds)


#load_img 功能
#将图像加载为PIL格式。
tf.keras.preprocessing.image.load_img(
    '../mian_directory/class_a/a_image_1.jpg', grayscale=False, color_mode="rgb", target_size=None, interpolation="nearest"
)
image = tf.keras.preprocessing.image.load_img('../mian_directory/class_a/a_image_1.jpg')
print(image)
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
print(input_arr)
# predictions = model.predict(input_arr)
# print(predictions)

#ImageDataGenerator 类
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
print(y_train)

epochs=10
num_classes=5
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)#将预测结果分为十个种类的onehot码子
y_test = np_utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs=epochs)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


