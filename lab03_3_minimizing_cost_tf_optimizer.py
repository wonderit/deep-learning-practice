import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# X and Y data
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(-3.0)

# Our hypothesis for linear model X * W
hypothesis = X * W

# Cost / Loss Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize : Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global varibles in the graph
sess.run(tf.global_variables_initializer())
for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
