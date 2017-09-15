import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

plt.style.use('ggplot')

""" Local minima/optima 
Gradient descent functions contain "minima" which is the lowest value
or point and maxima or the highest value. Local minima/maxima means the 
lowest/highest value of the local wave/trough
"""
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
x = np.linspace(-1, 1, 100)
hz = 10
cost = np.sin(hz*x)*np.exp(-x)
ax.plot(x, cost)
ax.set_ylabel('Cost')
ax.set_xlabel('Some Parameter')
plt.show()