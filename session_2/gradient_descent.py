import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

plt.style.use('ggplot')

# TODO: Understand the concepts especially cost function and get response from the forum
""" Local minima/optima 
Gradient descent functions contain "minima" which is the lowest value
or point and maxima or the highest value. Local minima/maxima means the 
lowest/highest value of the local wave/trough
"""
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
x = np.linspace(-1, 1, 200)
hz = 10
cost = np.sin(hz*x)*np.exp(-x)
# Get the difference between every value
gradient = np.diff(cost)
ax.plot(x, cost)
ax.set_ylabel('Cost')
ax.set_xlabel('Some Parameter')
# Number of iterations means the number of which the running/learning repeats
n_iterations = 500
cmap = plt.get_cmap('coolwarm')
c_norm = colors.Normalize(vmin=0, vmax=n_iterations)
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
""" init_p means the location of the nth(init_p) number in the linear space, 
down to the minima (if the gradient is negative)
"""
init_p = 120 # np.random.randint(len(x)*0.2, len(x)*0.8)
learning_rate = 1.0
# Implement a negative gradient of the function
for iter_i in range(n_iterations):
	# The dots must go down since it is decremented
	init_p -=learning_rate * gradient[int(init_p)]
	ax.plot(x[int(init_p)], cost[int(init_p)], 'ro', alpha=(iter_i + 1) / n_iterations, color=scalar_map.to_rgba(iter_i))
	ax.plot(x[int(init_p)])
plt.show()
