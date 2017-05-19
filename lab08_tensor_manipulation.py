# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

print('----------------------------------')

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

print('-----Shape, Rank, Axis------------')

t = tf.constant([1,2,3,4])
print(tf.shape(t).eval())


print('-----# rank : 2, Shape : ?------------')
t = tf.constant([[1,2],
                 [3,4]])
print(tf.shape(t).eval())

print('-----# rank : 4, Shape : ?------------')
t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(tf.shape(t).eval())

# [2,2] * [2,1]
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print(tf.matmul(matrix1, matrix2).eval())
# This is an error.. typo...
print((matrix1*matrix2).eval())

matrix1 = tf.constant([[3., 3.]])  # 2,1
matrix2 = tf.constant([[2.],[2.]]) # 1,2
print((matrix1+matrix2).eval())

print('------# Reduce Mean/Sum-----')

# becuas of int, answer is 1. not 1.5
print(tf.reduce_mean([1, 2], axis=0).eval())

x = [[1., 2.],
     [3., 4.]]


print(tf.reduce_mean(x).eval())
tf.reduce_mean(x, axis=0).eval()

tf.reduce_mean(x, axis=1).eval()
tf.reduce_mean(x, axis=-1).eval()
tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis=0).eval()
tf.reduce_sum(x, axis=-1).eval()
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()

x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()
tf.argmax(x, axis=1).eval()
tf.argmax(x, axis=-1).eval()

### Reshape, squeeze, expand_dims
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)


print(tf.reshape(t, shape=[-1, 3]).eval())
print(tf.reshape(t, shape=[-1, 1, 3]).eval())
print(tf.squeeze([[0], [1], [2]]).eval())
print(tf.expand_dims([0, 1, 2], 1).eval())

# One Hot
print('# one hot')
print(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())


t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print(tf.reshape(t, shape=[-1, 3]).eval())

# Casting
print('# Casting')
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

# Stack
print('# Stack')

x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=1).eval())


# Ones like and Zeros like
print('# Ones like and Zeros like')
x = [[0, 1, 2],
     [2, 1, 0]]

print tf.ones_like(x).eval()
print tf.zeros_like(x).eval()

# Zip
print 'Zip'
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)