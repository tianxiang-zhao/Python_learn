import numpy as np
import tensorflow as tf
a = np.array([
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],

              [
                  [-1, 5, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ]
            ])
b=np.array([[1,2,3],
            [4,5,6]])
print(np.argmax(a, axis=-1))
c=np.argmax(b, axis=-1)
c=c[...,tf.newaxis]
b=b[tf.newaxis,...]
print(c)
