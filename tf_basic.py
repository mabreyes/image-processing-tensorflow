import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

# Avoid console warnings by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" Create a linspace using numpy.linspace (start_val, end_val, 
number of contents in the array)
"""
x = np.linspace(-3.0, 3.0, 100)
print(x)
print(x.shape)
print(x.dtype)

# Create a linspace using tensorflow.linspace
x = tf.linspace(-3.0, 3.0, 100)
print(x)

# Inspect the graph
g = tf.get_default_graph()

# Create a session first before we can evaluate tensors
sess = tf.Session()
# Using sess without run won't work
computed_x = sess.run(x)
""" Alternatively we can use
compute_x = x.eval(session=sess)
We tell the tensor to evaluate itself using this session
"""

# Tell the session which graph to manage
sess = tf.Session(graph=g)

# Create a new graph
g2 = tf.Graph()

# Interactive session we need not to tell the eval about the session
sess = tf.InteractiveSession()
x.eval()

# sess.close() to close the session

# Use get_shape() to know the shape of your tensor
print(x.get_shape())
# A more friendly version uses []
print(x.get_shape().as_list())

# Gaussian/normal curve formula implementation
# f(x) = (1/(sqrt(2pi))exp(-((mean-x)^2/(2sigma(^2))))
# mean is mu
# standard deviation is sigma
# We use z as a representation for a z-test
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
						(2.0 * tf.pow(sigma, 2.0)))) * 
			(1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

# Evaluate the tensor for z
gaussian_graph = z.eval()
plt.plot(gaussian_graph)
plt.show()

# TODO: Understand more of this concept
# Creating a 2-D Gaussian Kernel
# (N, 1) x (1, N) inner dimensions should match, N are the result of matrix mult
# Store the number of values from our Gaussian curve
ksize = z.get_shape().as_list()[0]
"""Multiply the transposed matrix to the new shape with respect to the number of contents
of the Gaussian curve
"""
# tf.reshape arguments (tensor, shape, name)
# Multiply matrix to the transform of the matrix to get 2-D
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
plt.imshow(z_2d.eval())
plt.show()