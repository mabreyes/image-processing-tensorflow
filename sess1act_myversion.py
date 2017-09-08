from libs import utils
from skimage import data
from skimage.transform import resize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_files = [os.path.join('img_session1_assign', image_i)
               for image_i in os.listdir('img_session1_assign')
               if '.jpg' in image_i]

image_read = [plt.imread(image_i)
               for image_i in image_files]
image_read = [utils.imcrop_tosquare(image_i)
              for image_i in image_read]
image_read = [resize(image_i, (100, 100))
              for image_i in image_read]
images = np.array(image_read).astype(np.float32)

sess = tf.Session()

mean_images_4d = tf.reduce_mean(images, axis=0, keep_dims=True)
subtract = images - mean_images_4d

std = tf.sqrt(tf.divide(tf.reduce_mean(subtract * subtract, axis=0), images.shape[0]))
std = sess.run(std)
plt.show(plt.imshow(std, cmap='gray'))

# TODO: Identify error because convolved image is all black