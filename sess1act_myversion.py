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

image_read = [plt.imread(image_i)[..., :3]
               for image_i in image_files]
image_read = [utils.imcrop_tosquare(image_i)
              for image_i in image_read]
image_read = [resize(image_i, (100, 100))
              for image_i in image_read]
images = np.array(image_read).astype(np.float32)

sess = tf.Session()

mean_images_4d = tf.reduce_mean(images, axis=0, keep_dims=True)
subtract = images - mean_images_4d

# Do not use this type of deviation since it is not simplified to the nearest tens
# Using std_real (min, max) -48.3935 47.8217
# Using std_modified (min, max) -4.83935 4.78217
std_real = sess.run(tf.sqrt(tf.reduce_mean(subtract * subtract, axis=0) / images.shape[0]))
std_real_show = std_real / np.max(std_real)

std_mod = sess.run(tf.sqrt(tf.reduce_mean(subtract * subtract, axis=0)))
std_mod_show = std_mod / np.max(std_mod)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
axs[0].imshow(std_real_show)
axs[0].set_title('Real deviation ' + str(std_real_show.shape))
axs[1].imshow(std_mod_show)
axs[1].set_title('Modified deviation ' + str(std_mod_show.shape))
plt.show(fig.show)

# 0-1 normalization: (x - min(x)) / (max(x) - min(x))
normalize = sess.run(tf.divide(subtract, std_mod))
print(np.min(normalize), np.max(normalize))
# normalize_show = tf.divide(tf.subtract(normalize, np.min(normalize)), tf.subtract(np.max(normalize), np.min(normalize)))
normalize_show = tf.divide(normalize - np.min(normalize), np.max(normalize) - np.min(normalize))
# normalize_show = (normalize - np.min(normalize)) / (np.max(normalize) - np.min(normalize))
plt.imshow(utils.montage(normalize_show, 'normalized.png'))
plt.show()