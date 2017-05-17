import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# X and Y data
X = [1,2,3]
Y = [1,2,3]

# Set wrong model weights
W = tf.Variable(5.)

# Our hypothesis for linear model X * W
hypothesis = X * W

# Manual Gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# Cost / Loss Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize : Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost)

# Apply gradients
# train = optimizer.minimize(cost)
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session
sess = tf.Session()
# Initializes global varibles in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
