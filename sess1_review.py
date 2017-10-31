import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_files = [os.path.join('img_align_celeba_small', image_files_i)
               for image_files_i in os.listdir('img_align_celeba_small')
               if '.jpg' in image_files_i]

images = [plt.imread(image_files_i) for image_files_i in image_files]

plt.show(plt.imshow(images[0]))