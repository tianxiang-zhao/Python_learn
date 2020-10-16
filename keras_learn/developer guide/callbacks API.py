from tensorflow import keras
import tensorflow as tf

# 回调API
# 回调是一个对象，可以在训练的各个阶段（例如，在某个时期的开始或结束时，在单个批处理之前或之后等）执行操作。
#
# 您可以使用回调来：
#
# 每批培训后写TensorBoard日志以监控指标
# 定期将模型保存到磁盘
# 尽早停止
# 训练期间查看模型的内部状态和统计信息
# ...和更多

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# model.fit(dataset, epochs=10, callbacks=my_callbacks)

