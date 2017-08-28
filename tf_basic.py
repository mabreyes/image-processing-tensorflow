import tensorflow as tf 
import numpy as np 

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
computed_x = sess(x)
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