from libs import utils
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt 
from scipy.misc import imresize

plt.style.use('ggplot')

files = [os.path.join('img_align_celeba_small', file_i)
		 for file_i in os.listdir('img_align_celeba_small')
		 if '.jpg' in file_i]

read_files = [plt.imread(file_i)
			  for file_i in files]

data = np.array(read_files)

mean_data = np.mean(data, axis=0)
plt.imshow(mean_data.astype(np.uint8))
plt.show()

std_data = np.std(data, axis=0)
plt.imshow(std_data.astype(np.uint8))
plt.show()

""" Heat map
plt.imshow(np.mean(std_data, axis=0).astype(np.uint8))
plt.show()
"""