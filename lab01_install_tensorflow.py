import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print(tf.__version__)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# PlaceHolder

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

# EveryThing is Tensor
3 # a rank 0 tensor; this is a scalar with shape [ ]
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [1., 2., 3.]] # a rank 2 tensor; this is a matrix with shape [2, 3]
[[[1., 2., 3.]], [[1., 2., 3.]]] # a rank 3 tensor; with shape [2, 1, 3]