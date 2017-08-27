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